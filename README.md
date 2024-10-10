# 作业 3: 一个简单的 CUDA 渲染器

**总分：100 分**

![My Image](handout/teaser.jpg?raw=true)

## 概述

在这个作业中，你将编写一个使用 CUDA 并行渲染彩色圆圈的渲染器。虽然这个渲染器非常简单，但并行化渲染器需要你设计和实现能够在并行中高效构建和操作的数据结构。这是一个具有挑战性的作业，因此建议你尽早开始。**真的，建议你尽早开始。** 祝你好运！

## 环境设置

1. 你将在 Amazon Web Services (AWS) 上启用 GPU 的虚拟机上收集本作业的结果（即运行性能测试）。请按照 [cloud_readme.md](cloud_readme.md) 中的说明设置运行作业的机器。

2. 从课程的 Github 上下载作业启动代码：

`git clone https://github.com/stanford-cs149/asst3`

CUDA C++ Programming Guide [PDF 版本](http://docs.nvidia.com/cuda/pdf/CUDA_C_Programming_Guide.pdf) 或 [Web 版本](https://docs.nvidia.com/cuda/cuda-c-programming-guide/) 是学习如何在 CUDA 中编程的绝佳参考。网络上有大量的 CUDA 教程和 SDK 示例（只需搜索！）， [NVIDIA 开发者网站](http://docs.nvidia.com/cuda/) 上也有很多资源。特别的，你可能会喜欢免费的 Udacity 课程 [CUDA 并行编程简介](https://www.udacity.com/course/cs344)。

[CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#compute-capabilities) 中的表 G.1 是一个便捷的参考，提供了将在此作业中使用的 NVIDIA T4 GPU 所支持的线程块、线程块大小、共享内存等信息。NVIDIA T4 GPU 支持 CUDA compute capability 7.5。

对于 C++ 问题（例如 _virtual_ 关键字的含义是什么），[C++ Super-FAQ](https://isocpp.org/faq) 是一个很好的资源，以详细但易懂的方式解释问题（与许多 C++ 资源不同），并且由 C++ 的创建者 Bjarne Stroustrup 共同撰写。

### 警告

为了节省资源，当虚拟机在 15 分钟内 CPU 活动率低于 2% 时将自动停止。

这意味着如果你没有进行 CPU 密集型工作（例如编写代码），虚拟机将关闭。

因此，我们建议你在本地开发代码，然后手动将代码复制到虚拟机，或使用 git 将提交的代码拉取到虚拟机。使用 git 很好，因为你可以回退到代码的先前版本。

如果你之前没有设置过私人 git 仓库，这里有一些资源可以帮助你开始。确保 github 仓库是私有的，以确保你没有违反荣誉准则。

设置 git 的有用链接：

- [添加远程仓库](https://docs.github.com/en/get-started/getting-started-with-git/managing-remote-repositories) 以连接到你的私人仓库。
- [添加 SSH 密钥](https://docs.github.com/en/authentication/connecting-to-github-with-ssh/adding-a-new-ssh-key-to-your-github-account) 以设置 SSH 密钥。我们推荐不使用密码并使用默认名称 id_rsa 来完成此操作。

一旦你有了 SSH 密钥并知道如何连接到远程仓库，你需要做以下两件事来设置你的环境。

1. 将你的私钥复制到服务器的 .ssh 文件夹（即 .ssh 文件夹中的 id_rsa 文件）
2. 在服务器和本地创建一个名为 config 的文件，内容如下：

```
Host github.com
    HostName github.com
    User git
    IdentityFile ~/.ssh/id_rsa
```

你现在应该可以从服务器和本地拉取和推送提交了！

## 第一部分：CUDA 热身练习 1：SAXPY（5 分）

为了获得编写 CUDA 程序的一些实践经验，你的热身任务是用 CUDA 重新实现作业 1 中的 SAXPY 函数。该部分作业的启动代码位于作业仓库的`/saxpy`目录中。你可以通过在`/saxpy`目录下运行`make`和`./cudaSaxpy`来构建和运行 SAXPY CUDA 程序。

请完成`saxpy.cu`中`saxpyCuda`函数的 SAXPY 实现。在执行计算之前，你需要分配设备全局内存数组并将主机输入数组`X`、`Y`和`result`的内容复制到 CUDA 设备内存中。在 CUDA 计算完成后，结果必须被复制回主机内存。请参阅 Programmer's Guide（Web版本）第 3.2.2 节中`cudaMemcpy`函数的定义，或者查看作业起始代码中指向的有用教程。

作为你实现的一部分，在`saxpyCuda`中的 CUDA 内核调用周围添加计时器。添加计时器后，你的程序应对两个执行过程进行计时：

- 提供的起始代码包含计时器，用于测量将数据复制到 GPU、运行内核以及将数据复制回 CPU 的**整个过程**。

- 你还应该插入计时器来测量*运行内核所花费的时间*。 （不应包括 CPU 到 GPU 的数据传输时间或结果从 GPU 传输回 CPU 的时间。）

**在添加后者的计时代码时，你需要小心：**默认情况下，CUDA 内核在 GPU 上的执行是与在 CPU 上运行的主应用程序线程*异步*的。例如，如果你编写如下代码：

```
double startTime = CycleTimer::currentSeconds();
saxpy_kernel<<<blocks, threadsPerBlock>>>(N, alpha, device_x, device_y, device_result);
double endTime = CycleTimer::currentSeconds();
```

你会测量到一个非常快的内核执行时间！因为你只计时了 API 调用本身的成本，而不是在 GPU 上实际执行计算的成本。

因此，你需要在内核调用之后调用`cudaDeviceSynchronize()`以等待 GPU 上所有 CUDA 工作完成。该`cudaDeviceSynchronize()`调用会在 GPU 上所有先前的 CUDA 工作完成后返回。注意，在`cudaMemcpy()`之后不需要调用`cudaDeviceSynchronize()`来确保内存传输到 GPU 完成，因为·cudaMempy()·在我们使用它的条件下是同步的。（对于那些希望了解更多的人，请参阅[此文档](https://docs.nvidia.com/cuda/cuda-runtime-api/api-sync-behavior.html#api-sync-behavior__memcpy-sync)。）

```
double startTime = CycleTimer::currentSeconds();
saxpy_kernel<<<blocks, threadsPerBlock>>>(N, alpha, device_x, device_y, device_result);
cudaDeviceSynchronize();
double endTime = CycleTimer::currentSeconds();
```

请注意，在包括从 CPU 传输到 GPU 和结果从 GPU 返回的时间的测量中，不需要在最终计时器（在将数据返回到 CPU 的 `cudaMemcpy()` 调用之后）之前调用`cudaDeviceSynchronize()`，因为`cudaMemcpy()`在复制完成之前不会返回到调用线程。

**问题 1.** 与基于 CPU 的顺序 SAXPY 实现（回忆一下作业 1 中程序 5 的结果）相比，你观察到的性能如何？

**问题 2.** 比较并解释两个计时器集（仅对 kernal 执行进行计时，与对包括 kernal 执行、将数据移动到 GPU 并返回的整个过程进行计时）提供的结果之间的差异。观察到的带宽值是否与机器不同组件的报告带宽*大致*一致？（你应该使用网络查找 NVIDIA T4 GPU 的内存带宽。提示：<https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/tesla-t4/t4-tensor-core-datasheet-951643.pdf>。AWS 的内存总线预期带宽为 4 GB/s，这与 16 通道 [PCIe 3.0](https://en.wikipedia.org/wiki/PCI_Express) 的带宽不匹配。多个因素阻止了峰值带宽，包括 CPU 主板芯片组性能以及用作传输源的主机 CPU 内存是否“固定”——后者允许 GPU 直接访问内存，而无需经过虚拟内存地址转换。如果你有兴趣，可以在这里找到更多信息：<https://kth.instructure.com/courses/12406/pages/optimizing-host-device-data-communication-i-pinned-host-memory>）

## 第二部分：CUDA 热身练习 2：并行前缀和（10 分）

现在你已经熟悉了 CUDA 程序的基本结构和布局，作为第二个练习，你需要并行实现`find_repeats`函数。该函数接受一个整数列表`A`，返回所有使得`A[i] == A[i+1]`的索引`i`的列表。

例如，给定数组`{1,2,2,1,1,1,3,5,3,3}`，你的程序应输出数组`{1,3,4,8}`。

### 排除前缀和

我们希望你通过首先实现并行的排除前缀和（exclusive prefix dum）操作来实现`find_repeats`。

排除前缀和接受一个数组`A`并生成一个新的数组`output`，`output[i]`处的值为数组`A`中所有索引小于 `i` 的元素的和（不包括`A[i]`）。例如，给定数组`A={1,4,6,8,2}`，排除前缀和的输出为 `output={0,1,5,11,19}`。

以下“类 C”代码是 scan 的迭代版本。在伪代码中，我们使用`parallel_for`表示可能并行的循环。这是我们在课堂上讨论过的算法：http://cs149.stanford.edu/fall23/lecture/dataparallel/slide_17

```
void exclusive_scan_iterative(int* start, int* end, int* output) {

    int N = end - start;
    memmove(output, start, N*sizeof(int));
    
    // upsweep阶段
    for (int two_d = 1; two_d <= N/2; two_d*=2) {
        int two_dplus1 = 2*two_d;
        parallel_for (int i = 0; i < N; i += two_dplus1) {
            output[i+two_dplus1-1] += output[i+two_d-1];
        }
    }

    output[N-1] = 0;

    // downsweep阶段
    for (int two_d = N/2; two_d >= 1; two_d /= 2) {
        int two_dplus1 = 2*two_d;
        parallel_for (int i = 0; i < N; i += two_dplus1) {
            int t = output[i+two_d-1];
            output[i+two_d-1] = output[i+two_dplus1-1];
            output[i+two_dplus1-1] += t;
        }
    }
}
```

我们希望你使用此算法在 CUDA 中实现并行前缀和。你必须在`scan/scan.cu`中实现 `exclusive_scan` 函数。你的实现将包含主机代码和设备代码。实现将需要多个 CUDA kernel launch（上面伪代码中每个 parallel_for 对应一个）。

**注意：** 在起始代码中，上述参考 scan 实现假设输入数组的长度（`N`）是 2 的幂。在 `cudaScan` 函数中，我们通过在 GPU 上分配相应缓冲区时将输入数组长度舍入到下一个 2 的幂来解决此问题。但是，代码只会从 GPU 缓冲区复制回 CPU 缓冲区的`N`个元素。这一事实应简化你的 CUDA 实现。

编译生成 `cudaScan` 二进制文件。命令行用法如下：

```
Usage: ./cudaScan [options] 

Program Options:
  -m  --test <TYPE>      Run specified function on input.  Valid tests are: scan, find_repeats (default: scan)
  -i  --input <NAME>     Run test on given input type. Valid inputs are: ones, random (default: random)
  -n  --arraysize <INT>  Number of elements in arrays
  -t  --thrust           Use Thrust library implementation
  -?  --help             This message
```

### 使用前缀和实现“查找重复项”

编写完`exclusive_scan`之后，在`scan/scan.cu`中实现`find_repeats`函数。这将涉及编写更多的设备代码，以及对`exclusive_scan()`的一个或多个调用。你的代码应将重复元素列表写入提供的输出指针（在设备内存中），然后返回输出列表的大小。

调用`exclusive_scan`实现时，请记住`start`数组的内容会被复制到`output`数组中。此外，传递给`exclusive_scan`的数组应位于`device`内存中。

**评分：** 我们将测试你的代码在随机输入数组上的正确性和性能。

作为参考，下面提供了一个扫描得分表，显示了 K80 GPU 上简单 CUDA 实现的性能。要检查你的`scan`和`find_repeats`实现的正确性和性能得分，分别运行**`./checker.pl scan`**和 **`./checker.pl find_repeats`**。这样做会生成一个如下所示的参考表；你的得分完全基于代码的性能。为了获得满分，你的代码性能必须在参考解决方案的 20% 以内。

```
-------------------------
Scan Score Table:
-------------------------
-------------------------------------------------------------------------
| Element Count   | Ref Time        | Student Time    | Score           |
-------------------------------------------------------------------------
| 1000000         | 0.766           | 0.143 (F)       | 0               |
| 10000000        | 8.876           | 0.165 (F)       | 0               |
| 20000000        | 17.537          | 0.157 (F)       | 0               |
| 40000000        | 34.754          | 0.139 (F)       | 0               |
-------------------------------------------------------------------------
|                                   | Total score:    | 0/5             |
-------------------------------------------------------------------------
```

这一部分的作业主要是让你更多地练习编写 CUDA 和用数据并行方式思考，而不是性能调优代码。在这一部分作业中获得满分不需要太多（或确实不需要）性能调优，只需将算法伪代码直接移植到 CUDA 即可。然而，有一个技巧：扫描的简单实现可能会为伪代码中的每次并行循环迭代启动 N 个 CUDA 线程，并在内核中使用条件执行来确定哪些线程实际上需要工作。这种解决方案不会有好的性能！（考虑upsweep阶段最外层循环的最后一次迭代，其中只有两个线程会工作！）满分解决方案只会为最内层并行循环的每次迭代启动一个 CUDA 线程。

**测试工具：** 默认情况下，测试工具在相同的伪随机生成数组上运行，以帮助调试。你可以传递参数`-i random`在随机数组上运行——我们在评分时会这样做。我们鼓励你提出替代的输入来帮助评估你的程序。你还可以使用`-n <size>`选项更改输入数组的长度。

参数`--thrust`将使用 [Thrust 库](http://thrust.github.io/) 实现的 [排除扫描](http://thrust.github.io/doc/group__prefixsums.html)。**最多可以获得两分的额外奖励，前提是你能创建一个与 Thrust 竞争的实现。**

## 第三部分：简单的圆形渲染器（85 分）

现在进入正式部分！

作业起始代码的`/render`目录包含一个绘制彩色圆形的渲染器实现。构建代码，并运行以下命令来运行渲染器：`./render -r cpuref rgb`。程序将输出一个包含三个圆形的图像`output_0000.ppm`。现在运行命令行`./render -r cpuref snow`来运行渲染器。此时输出图像将是飘落的雪花。在 OSX 上可以通过预览直接查看 PPM 图像。在 Windows 上，你可能需要下载一个查看器。

注意：你也可以使用`-i`选项将渲染器输出发送到显示器而不是文件。（在雪景情况下，你会看到降雪动画。）但是，要使用交互模式，你需要能够设置 X-windows 转发到本地机器。（[参考1](http://atechyblog.blogspot.com/2014/12/google-cloud-compute-x11-forwarding.html) 或 [参考2](https://stackoverflow.com/questions/25521486/x11-forwarding-from-debian-on-google-compute-engine) 可能会有所帮助。）

作业起始代码包含两个版本的渲染器：一个串行的、单线程的 C++ 参考实现，位于`refRenderer.cpp`中，以及一个*不正确的*并行 CUDA 实现，位于`cudaRenderer.cu`中。

### 渲染器概述

我们鼓励你通过检查`refRenderer.cpp`中的参考实现来熟悉渲染器代码库的结构。`setup`方法在渲染第一帧之前调用。在你的 CUDA 加速渲染器中，这个方法可能包含所有的渲染器初始化代码（分配缓冲区等）。`render`方法在每一帧中调用，负责将所有圆绘制到输出图像中。渲染器的另一个主要函数`advanceAnimation`也在每帧调用一次。它更新圆的位置和速度。你不需要在这个作业中修改`advanceAnimation`。

渲染器接受一个圆的数组（3D 位置、速度、半径、颜色）作为输入。渲染每一帧的基本串行算法是：

```
Clear image
for each circle
    update position and velocity
for each circle
    compute screen bounding box
    for all pixels in bounding box
        compute pixel center point
        if center point is within the circle
            compute color of circle at point
            blend contribution of circle into image for this pixel
```

下图说明了使用点在圆内测试（point-in-circle tests）计算圆像素覆盖的基本算法。请注意，只有当像素的中心位于圆内时，圆才会对输出像素产生颜色贡献。

![点在圆内测试](handout/point_in_circle.jpg?raw=true "计算圆对输出图像贡献的简单算法：测试圆边界框内的所有像素是否被覆盖。对于边界框内的每个像素，如果其中心点（黑点）包含在圆内，则认为该像素被圆覆盖。圆圈内的像素中心被涂成红色。仅针对被覆盖的像素计算圆圈对图像的贡献。")

渲染器的一个重要细节是它渲染**半透明**的圆。因此，任何一个像素的颜色都不是单个圆的颜色，而是所有重叠在该像素上的半透明圆的贡献的混合结果（注意上面伪代码中的“混合贡献”（blend contribution）部分）。渲染器通过红（R）、绿（G）、蓝（B）和不透明度（alpha）值的 4 元组（RGBA）表示圆的颜色。Alpha = 1 对应完全不透明的圆。Alpha = 0 对应完全透明的圆。要在颜色为`(P_r, P_g, P_b)`的像素上绘制颜色为`(C_r, C_g, C_b, C_alpha)`的半透明圆，渲染器使用以下数学公式：

<pre>
   result_r = C_alpha * C_r + (1.0 - C_alpha) * P_r
   result_g = C_alpha * C_g + (1.0 - C_alpha) * P_g
   result_b = C_alpha * C_b + (1.0 - C_alpha) * P_b
</pre>

注意，合成是不可交换的（对象 X 在 Y 之上与对象 Y 在 X 之上看起来不同），因此渲染器必须按应用程序提供的顺序绘制圆。（你可以假设应用程序按深度顺序提供圆。）例如，考虑下图，其中一个蓝色圆绘制在一个绿色圆之上，而绿色圆绘制在一个红色圆之上。在左图中，圆按正确的顺序绘制到输出图像中。在右图中，圆的绘制顺序不同，输出图像看起来不正确。

![顺序](handout/order.jpg?raw=true "渲染器必须小心生成与按应用程序提供的顺序逐个绘制所有圆时生成的输出相同的图像。")

### CUDA 渲染器

在熟悉了参考代码中实现的圆形渲染算法后，现在研究`cudaRenderer.cu`中提供的渲染器的 CUDA 实现。你可以使用`--renderer cuda`（或 `-r cuda`）选项运行 CUDA 渲染器。

提供的 CUDA 实现在所有输入圆上并行化计算，每个 CUDA 线程分配一个圆。虽然这个 CUDA 实现完整地实现了圆形渲染器的数学原理，但它包含几个主要错误，你要在本作业中修复这些错误。具体来说：当前实现未确保图像更新是一个原子操作，也未维护图像更新的必要顺序（顺序要求将在下文描述）。

### 渲染器要求

你的并行 CUDA 渲染器实现必须维护串行实现中自动维护的两个不变量。

1. **原子性：**所有图像更新操作必须是原子的。临界区包括读取四个 32 位浮点值（像素的 RGBA 颜色）、混合当前圆的贡献与当前图像值，然后将像素的颜色写回内存。
2. **顺序：**渲染器必须按*圆输入顺序*对图像像素进行更新。也就是说，如果圆 1 和圆 2 都对像素 P 有贡献，则圆 1 对 P 的任何图像更新必须在圆 2 的更新之前应用于图像。如上所述，维护顺序要求可以正确渲染透明圆。（它对图形系统还有一些其他好处。如果好奇，可以与 Kayvon 谈谈。）**一个关键观察是，顺序定义仅指定对同一像素的更新顺序。** 因此，如下图所示，不对同一像素有贡献的不同圆之间不存在顺序要求。这些圆可以独立处理。

![依赖关系](handout/dependencies.jpg?raw=true "圆 1、圆 2 和圆 3 的贡献必须按提供给渲染器的顺序应用于重叠的像素。")

由于提供的 CUDA 实现不满足这些要求，因此在 RGB 和圆形场景上运行 CUDA 渲染器实现时，可以看到没有正确遵循顺序或原子性的结果。你会看到结果图像中有水平条纹，如下图所示。这些条纹会在每帧中变化。

![顺序错误](handout/bug_example.jpg?raw=true "由于帧缓冲区更新缺乏原子性导致输出中的错误（注意图像底部的条纹）。")

### 你需要做什么

**你的任务是尽可能编写最快且正确的 CUDA 渲染器实现**。你可以采取任何你认为合适的方法，但你的渲染器必须遵守上述的原子性和顺序要求。不满足这两个要求的解决方案在作业的第 3 部分将最多只能获得 12 分。我们已经给了你这样一个解决方案！

一个好的起点是通读`cudaRenderer.cu`并确信它*不*满足正确性要求。特别是，查看`CudaRenderer:render`是如何启动 CUDA 核心`kernelRenderCircles`的。（`kernelRenderCircles`是所有工作发生的地方。）为了直观地看到违反上述两个要求的效果，使用`make`编译程序。然后运行`./render -r cuda rand10k`，这应该会显示包含 10K 个圆的图像，如上图底部所示。通过运行`./render -r cpuref rand10k`将此（不正确的）图像与串行代码生成的图像进行比较。

我们建议你：

1. 首先重写 CUDA 起始代码实现，使其在并行运行时逻辑上正确（我们推荐一种不需要锁或同步的方法）。
2. 然后确定你的解决方案存在的性能问题。
3. 这时，对作业的真正思考开始了……（提示：提供给你的`circleBoxTest.cu_inl`中的圆与矩形框相交测试（circle-intersects-box tests）是你的朋友。鼓励你使用这些子例程。）

以下是`./render`的命令行选项：

```
Usage: ./render [options] scenename
Valid scenenames are: rgb, rgby, rand10k, rand100k, biglittle, littlebig, pattern,
                      bouncingballs, fireworks, hypnosis, snow, snowsingle
Program Options:
  -r  --renderer <cpuref/cuda>  Select renderer: ref or cuda (default=cuda)
  -s  --size  <INT>             Make rendered image <INT>x<INT> pixels (default=1024)
  -b  --bench <START:END>       Run for frames [START,END)   (default [0,1))
  -f  --file  <FILENAME>        Output file name (FILENAME_xxxx.ppm)
  -c  --check                   Check correctness of CUDA output against CPU reference
  -i  --interactive             Render output to interactive display
  -?  --help                    This message
```

**检查代码：**为了检测程序的正确性，`render`有一个方便的`--check`选项。此选项会运行参考 CPU 渲染器的串行版本以及你的 CUDA 渲染器，然后比较生成的图像以确保正确性。还会打印你的 CUDA 渲染器实现所花的时间。

我们提供了五个圆形数据集，你将根据它们进行评分。然而，为了获得满分，你的代码必须通过我们所有的正确性测试。要检查你的代码的正确性和性能分数，请在`/render`目录中运行**`./checker.py`**（注意扩展名为 .py）。如果你在起始代码上运行它，程序将打印如下表格，以及整个测试集的结果：

```
------------
Score table:
------------
--------------------------------------------------------------------------
| Scene Name      | Ref Time (T_ref) | Your Time (T)   | Score           |
--------------------------------------------------------------------------
| rgb             | 0.2321           | (F)             | 0               |
| rand10k         | 5.7317           | (F)             | 0               |
| rand100k        | 25.8878          | (F)             | 0               |
| pattern         | 0.7165           | (F)             | 0               |
| snowsingle      | 38.5302          | (F)             | 0               |
| biglittle       | 14.9562          | (F)             | 0               |
--------------------------------------------------------------------------
|                                    | Total score:    | 0/72            |
--------------------------------------------------------------------------
```

注意：在某些运行中，你*可能*会在某些场景中得分，因为提供的渲染器的运行时是非确定性的，有时它可能是正确的。这并不改变当前 CUDA 渲染器通常是不正确的事实。

“Ref time” 是参考解决方案在你当前机器上的性能（在提供的`render_ref`可执行文件中）。“Your time” 是你当前 CUDA 渲染器解决方案的性能，其中`(F)`表示不正确的解决方案。你的分数将取决于你的实现相对于这些参考实现的性能（详见评分指南）。

除了代码外，我们还希望你提交一个清晰的高水平描述，说明你的实现是如何工作的，以及你如何得出这个解决方案。具体来说，介绍你在此过程中尝试的方法，以及你如何确定优化代码的方法（例如，你进行了哪些测量来指导你的优化工作？）。

你应该在报告中提到的工作方面包括：

1. 在报告顶部注明两位合作者的名字和 SUNet id。
2. 复制为你的解决方案生成的得分表，并指定你运行代码的机器。
3. 描述你如何分解问题以及如何将工作分配给 CUDA 线程块和线程（甚至可能是 warp）。
4. 描述你的解决方案中出现同步的位置。
5. 你采取了哪些步骤来减少通信需求（例如，同步或主内存带宽需求）？
6. 简要描述你是如何得出最终解决方案的。你在此过程中尝试了哪些其他方法？它们有什么问题？

### 评分指南

- 作业报告占 7 分。
- 你的实现占 72 分。它们被平均分配到每个场景的 12 分，如下所示：
  - 每个场景 2 分的正确性分数。
  - 每个场景 10 分的性能分数（只有在解决方案正确的情况下才能获得）。你的性能将相对于提供的基准参考渲染器 T<sub>ref</sub> 的性能进行评分：
    - 对于时间 (T) 是 T<sub>ref</sub> 的 10 倍的解决方案，不会给予性能分数。
    - 对于在优化解决方案 20% 以内的解决方案（T < 1.20 * T<sub>ref</sub>），将给予全额性能分数。
    - 对于其他 T 值（1.20 T<sub>ref</sub> <= T < 10 * T<sub>ref</sub>），你的性能分数将在 1 到 10 的范围内按以下公式计算：`10 * T_ref / T`。
- 你的实现在班级排行榜上的表现占最后 6 分。排行榜的提交和评分细节将在随后的 Ed 帖子中详细说明。
- 对于性能显著高于要求的解决方案，最多可获得 5 分的额外加分（由导师决定）。你的报告必须清楚地解释你的方法。
- 对于高质量的仅 CPU 并行渲染器实现，并且能很好地利用所有内核和内核的 SIMD 向量单元，最多可获得 5 分的额外加分（由导师决定）。可以使用任何工具（例如，SIMD 内联函数、ISPC、pthread）。为了获得加分，你应该分析你的 GPU 和 CPU 解决方案的性能，并讨论实现选择差异的原因。

因此，该项目的总分如下：

- 第 1 部分（5 分）
- 第 2 部分（10 分）
- 第 3 部分报告（7 分）
- 第 3 部分实现（72 分）
- 第 3 部分排行榜（6 分）
- 可能的**额外**加分（最多 10 分）

## 作业提示和建议

以下是从前几年的作业中整理的一些提示和建议。请注意，有多种实现渲染器的方法，并非所有提示都适用于你的方法。

- 在这个作业中有两个潜在的并行性轴。一个轴是*像素的并行性*，另一个是*圆的并行性*（前提是要尊重重叠圆的顺序要求）。解决方案需要在计算的不同部分利用这两种并行性。
- 在`circleBoxTest.cu_inl`中提供的圆与矩形框相交测试（circle-intersects-box tests）是你的朋友。鼓励你使用这些子例程。
- 在`exclusiveScan.cu_inl`中提供的共享内存前缀和操作对你完成这个作业可能很有价值（并非所有解决方案都会选择使用它）。请参阅[前缀和的简单描述](http://thrust.github.io/doc/group__prefixsums.html)。我们已经提供了在共享内存中对 **2 的幂大小**的数组进行排除前缀和的实现。**提供的代码不适用于非 2 的幂大小的输入，并且需要线程块中的线程数量与数组大小相同。请阅读代码中的注释。**
- 如果你愿意，可以在实现中使用[Thrust 库](http://thrust.github.io/)。Thrust 并不是实现优化的 CUDA 参考实现性能所必需的。有一种流行的解决问题的方法使用我们提供的共享内存前缀和实现。还有另一种流行的方法使用 Thrust 库中的前缀和例程。两者都是有效的解决策略。
- 渲染器中是否存在数据重用？可以做些什么来利用这种重用？
- 由于没有 CUDA 语言原语可以原子地执行图像更新操作，你将如何确保图像更新的原子性？构建全局内存原子操作的锁是一种解决方案，但请记住，即使图像更新是原子的，更新也必须按照要求的顺序进行。**我们建议你首先考虑在并行解决方案中确保顺序，然后再考虑你的解决方案中（如果仍然存在）原子性问题。**
- 如果你有空闲时间，可以尽情制作自己的场景！

### 捕捉 CUDA 错误 ###

默认情况下，如果你访问数组越界、分配了过多内存或导致其他错误，CUDA 通常不会通知你；相反，它会默默地失败并返回一个错误代码。你可以使用以下宏（可以随意修改）来包装 CUDA 调用：

```
#define DEBUG

#ifdef DEBUG
#define cudaCheckError(ans) { cudaAssert((ans), __FILE__, __LINE__); }
inline void cudaAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr, "CUDA Error: %s at %s:%d\n", 
        cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}
#else
#define cudaCheckError(ans) ans
#endif
```

注意，你可以取消定义 DEBUG 以在代码正确后禁用错误检查，从而提高性能。

然后你可以包装 CUDA API 调用来处理返回的错误，如下所示：

```
cudaCheckError( cudaMalloc(&a, size*sizeof(int)) );
```

注意，你不能直接包装内核启动。相反，它们的错误将在你下一个包装的 CUDA 调用时被捕捉：

```
kernel<<<1,1>>>(a); // 假设内核导致了一个错误！
cudaCheckError( cudaDeviceSynchronize() ); // 错误在这行被打印
```

所有 CUDA API 函数，如`cudaDeviceSynchronize`、`cudaMemcpy`、`cudaMemset`等都可以被包装。

**重要提示：**如果一个 CUDA 函数先前出错但未被捕捉到，那么即使包装了不同的函数，该错误也会在下一个错误检查中显示。例如：

```
...
line 742：cudaMalloc(&a, -1); // 执行，然后继续
line 743：cudaCheckError(cudaMemcpy(a,b)); // 打印 "CUDA Error: out of memory at cudaRenderer.cu:743"
...
```

因此，在调试时，建议你包装**所有** CUDA API 调用（至少是在你编写的代码中）。

（来源：改编自[这篇 Stack Overflow 帖子](https://stackoverflow.com/questions/14038589/what-is-the-canonical-way-to-check-for-errors-using-the-cuda-runtime-api)）

## 提交说明

请使用 Gradescope 提交你的作业。如果你和合作伙伴一起工作，请记得在 Gradescope 上标记你的合作伙伴。

1. **请提交名为`writeup.pdf`的报告文件。**
2. **请运行`sh create_submission.sh`生成一个 zip 文件并提交到 Gradescope。** 请注意，这将会在你的代码目录中运行`make clean`，所以你需要再次运行`make`来运行你的代码。如果脚本报错提示“权限被拒绝”，你应该运行`chmod +x create_submission.sh`，然后重新运行脚本。

我们的评分脚本将重新运行检测代码，使我们能验证你的分数与`writeup.pdf`中提交的分数匹配。我们可能还会尝试在其他数据集上运行你的代码以进一步检查其正确性。