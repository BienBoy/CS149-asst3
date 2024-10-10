# 实验报告

实验环境：WSL2；GeForce MX350；Cuda 12.6

## 第一部分：CUDA 热身练习 1：SAXPY

实验结果：

![SAXPY](https://cdn.jsdelivr.net/gh/BienBoy/images/images/2024%2F10%2F07%2F15-50-09-SAXPY.png)

相比基于 CPU 的实现，性能明显下降。这是由于 SAXPY 属于 I/O 密集型任务，计算量较小，主要的时间耗费在数据的转移。

## 第二部分：CUDA 热身练习 2：并行前缀和

![scan](https://cdn.jsdelivr.net/gh/BienBoy/images/images/2024%2F10%2F10%2F13-43-16-scan.png)

![find_repeats](https://cdn.jsdelivr.net/gh/BienBoy/images/images/2024%2F10%2F10%2F13-43-19-find_repeats.png)

## 第三部分：简单的圆形渲染器

渲染时，圆渲染的先后关系会影响正确性。初始错误实现中，`CudaRenderer::render()` 中的 kernal launch 是为每个圆分配一个线程并行渲染，渲染的先后关系及原子性要求均无法满足。

渲染器两个潜在的并行性轴：像素的并行性和圆的并行性。为每个圆分配一个线程并行渲染无法满足要求，那就改为为每个像素分配一个线程。每个线程内按顺序渲染圆。由此得到以下正确实现：

```c++
__global__ void kernelRenderPixels() {
    int pixelX = blockDim.x * blockIdx.x + threadIdx.x;
    int pixelY = blockDim.y * blockIdx.y + threadIdx.y;
    short imageWidth = cuConstRendererParams.imageWidth;
    short imageHeight = cuConstRendererParams.imageHeight;
    if (pixelX >= imageWidth || pixelY >= imageHeight)
        return;

    float invWidth = 1.f / imageWidth;
    float invHeight = 1.f / imageHeight;
    float2 pixelCenterNorm = make_float2(invWidth * (static_cast<float>(pixelX) + 0.5f),
                                         invHeight * (static_cast<float>(pixelY) + 0.5f));
    float4* imgPtr = (float4*)(&cuConstRendererParams.imageData[4 * (pixelY * imageWidth + pixelX)]);

    int numCircles = cuConstRendererParams.numCircles;
    for (int i = 0; i < numCircles; ++i) {
        float3 p = *(float3*)(&cuConstRendererParams.position[3 * i]);
        shadePixel(i, pixelCenterNorm, p, imgPtr);
    }
}

void CudaRenderer::render() {
    dim3 blockDim(16, 16);
    dim3 gridDim(
        (image->width + blockDim.x - 1) / blockDim.x,
        (image->height + blockDim.y - 1) / blockDim.y);
    kernelRenderPixels<<<gridDim, blockDim>>>();
    cudaDeviceSynchronize();
}
```

以上实现正确，但性能较差，原因在于每个线程内都顺序计算每个圆对单个像素颜色的贡献，没有充分利用圆的并行性。

考虑实际情况，对于一个圆来说，图片中的大部分像素可能都不需要渲染。优化方法是每个 block 内线程使用共享内存共同配合完成圆圈的初步筛选。由于每个线程负责一个像素，一个 block 内的所有线程负责的则是图片的一个矩形区域，如果一个圆和这个矩形区域不相交，则说明该区域内的像素均不需要渲染，因此，在渲染每个像素时，只需遍历与该矩形区域相交的圆即可。

计算与子区域有相交的圆，可以采用和第二部分类似的算法，使用并行前缀和实现。

```c++
__global__ void kernelRenderPixels() {
    uint idx = blockDim.x * threadIdx.y + threadIdx.x;
    uint pixelX = blockDim.x * blockIdx.x + threadIdx.x;
    uint pixelY = blockDim.y * blockIdx.y + threadIdx.y;
    short imageWidth = (short)cuConstRendererParams.imageWidth;
    short imageHeight = (short)cuConstRendererParams.imageHeight;

    float invWidth = 1.f / (float)imageWidth;
    float invHeight = 1.f / (float)imageHeight;
	
    // 线程块负责的矩形区域
    uint boxL = blockDim.x * blockIdx.x;
    uint boxR = boxL + blockDim.x < imageWidth ? boxL + blockDim.x : imageWidth;
    uint boxB = blockDim.y * blockIdx.y;
    uint boxT = boxB + blockDim.y < imageHeight ? boxB + blockDim.y : imageHeight;

    float boxLNorm = (float)boxL * invWidth;
    float boxRNorm = (float)boxR * invWidth;
    float boxBNorm = (float)boxB * invHeight;
    float boxTNorm = (float)boxT * invHeight;

    __shared__ uint flag[BLOCKSIZE];
    __shared__ uint presum[BLOCKSIZE];
    __shared__ uint scratch[BLOCKSIZE * 2];
    __shared__ uint circles[BLOCKSIZE];

    float2 pixelCenterNorm = make_float2(invWidth * (static_cast<float>(pixelX) + 0.5f),
                                         invHeight * (static_cast<float>(pixelY) + 0.5f));
    float4* imgPtr = (float4*)(&cuConstRendererParams.imageData[4 * (pixelY * imageWidth + pixelX)]);

    int numCircles = cuConstRendererParams.numCircles;
    for (int i = 0; i < numCircles; i += BLOCKSIZE) {
        uint circleIdx = i + idx;
        // 本线程负责检查的圆的序号
        // 检查圆是否与该区域有交叉
        if (circleIdx < numCircles) {
            float3 p = *(float3*)(&cuConstRendererParams.position[3 * circleIdx]);
            flag[idx] = circleInBoxConservative(p.x, p.y,
                cuConstRendererParams.radius[circleIdx],
                boxLNorm, boxRNorm, boxTNorm, boxBNorm);
        } else {
            flag[idx] = 0;
        }
        __syncthreads();

        // 计算flag的前缀和
        sharedMemExclusiveScan((int)idx, flag, presum, scratch, BLOCKSIZE);
        __syncthreads();

        // 获取所有可能和区域相交的圆
        if (flag[idx])
            circles[presum[idx]] = circleIdx;
        __syncthreads();

        // 进行渲染
        if (pixelX < imageWidth && pixelY < imageHeight) {
            uint num = presum[BLOCKSIZE - 1] + flag[BLOCKSIZE - 1]; // 可能和区域相交的圆的数量
            for (int j = 0; j < num; ++j) {
                float3 p = *(float3*)(&cuConstRendererParams.position[3 * circles[j]]);
                shadePixel((int)circles[j], pixelCenterNorm, p, imgPtr);
            }
        }
    }
}
```

![render](https://cdn.jsdelivr.net/gh/BienBoy/images/images/2024%2F10%2F10%2F13-43-23-render.png)