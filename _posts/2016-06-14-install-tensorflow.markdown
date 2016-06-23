---
layout: post
title:  "Installation of Tensorflow"
date:   2016-06-14 21:20:52 +0800
categories: Miscellany
---


# What I need:
---
- Tensorflow with GPU support
- Compile from source code for potential hacking into the code


# What I have:
---
- Ubuntu 16.04
- GCC 5.3.1
- Xeon E5 1620
- Nvidia Geforce Titan X


# What I get:
---
- Tensorflow 0.9rc from source code
- GPU support enabled
- CUDA 7.5
- cuDNN v5


Tensorflow is now only compatible with Linux and MacOS right now. For some reason, Windows servers are pervasive in my case, so the installation process has deviated from the right path from the beginning. I've struggled for quite some time to try to find a way to install tensorflow in a linux virtual machine environment. But for all the hypervisor softwares that I tried, like VirtualBox, VMware Player and Hyper-V, GPUs are invisible to the virtual machine. There are hypervisors in Windows like VMware vSphere that can passthrough GPU to virtual machine, while I cannot afford it. So do not avoid the trouble of installing a dual-boot system, as you will get more trouble in linux virtual machines. Of course you can still dip your toes with tensorflow in VMs if you don't want GPU support.  

### Starting with GPU support
Regardless of all the compatibility issues and warnings during the process, I built my GPU drivers and CUDA libraries with the newest version of `Ubuntu 16.04` and `gcc 5.3.1`, and successfully installed `CUDA 7.5` and `cuDNN v5`. I got a lot of help from Dr Donald Kinghorn [^1].

Tensorflow requires NVIDIA's Cuda Toolkit (>=7.0) and cuDNN (>=2.0), with a GPU card with NVIDIA Compute Capability >=3.0. A convenient table to check out the compute capability of your GPU can be found on [Wikipedia](https://en.wikipedia.org/wiki/CUDA) and [NVidia Website](https://developer.nvidia.com/cuda-gpus). Typically Titan X has compute capability of 5.2 and highest available compute capability for now is 6.1 from GeForce GTX 1080/1070. 

# GPU Driver

Download and install newest GPU driver:
{% highlight shell %}
$ wget http://us.download.nvidia.com/XFree86/Linux-x86_64/361.45.11/NVIDIA-Linux-x86_64-361.45.11.run
$ chmod a+x NVIDIA-Linux-x86_64-361.45.11.run
{% endhighlight %}
You will have to stop any X servers before you install the driver:
{% highlight shell %}
$ sudo service lightdm stop
$ sudo service gdm stop
$ sudo service mdm stop
{% endhighlight %}
Install the driver:
{% highlight shell %}
$ ./NVIDIA-Linux-x86_64-361.45.11.run
{% endhighlight %}

To check if the driver is installed, use the NVIDIA System Management Interface to check you GPU status:

~~~bash
$ nvidia-smi
~~~

# CUDA 7.5

Download CUDA 7.5 from [NVidia official website](https://developer.nvidia.com/cuda-downloads). In my case, `linux`, `x86_64`, `Ubuntu`, `15.04` and `runfile` are selected despite I have Ubuntu 16.04. From the folder the runfile is downloaded into:

~~~bash
$ chmod a+x cuda_7.5.18_linux.run
$ sudo ./cuda_7.5.18_linux.run --override
~~~

CUDA 7.5 is compatible with up to `gcc 4.9` while I have `gcc 5.3.1` by default, and I can use `--override` to surpress the warning of `unsupported compiler`.  

As newest GPU driver is installed, choose **NOT** install the older version driver. Also make sure the symbolic link at `/usr/local/cuda` is installed as it is required by tensorflow. 

~~~
Do you accept the previously read EULA? (accept/decline/quit): accept 
You are attempting to install on an unsupported configuration. Do you wish to continue? ((y)es/(n)o) [ default is no ]: y
Install NVIDIA Accelerated Graphics Driver for Linux-x86_64 352.39? ((y)es/(n)o/(q)uit): n
Install the CUDA 7.5 Toolkit? ((y)es/(n)o/(q)uit): y
Enter Toolkit Location [ default is /usr/local/cuda-7.5 ]:\
Do you want to install a symbolic link at /usr/local/cuda? ((y)es/(n)o/(q)uit): y
Install the CUDA 7.5 Samples? ((y)es/(n)o/(q)uit): y
Enter CUDA Samples Location [ default is /home/minint ]: 
~~~

Setup environment variables in `~/.bashrc`

~~~bash
export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/usr/local/cuda/lib64"
export CUDA_HOME=/usr/local/cuda
~~~

# cuDNN v5
Download [cuDNN v5](https://developer.nvidia.com/rdp/cudnn-download) for CUDA 7.5. Choose cuDNN v5 Library for Linux to start download and change the file extension from `.solitairetheme8` to `.tgz`. ~~(Seriously? why? NVIDIA? why?)~~

~~~bash
$ tar fvxz cudnn-7.5-linux-x64-v5.0-ga.tgz
$ cd cuda
$ sudo cp include/cudnn.h /usr/local/cuda/include
$ sudo cp lib64/libcudnn* /usr/local/cuda/lib64
$ sudo chmod a+r /usr/local/cuda/include/cudnn.h /usr/local/cuda/lib64/libcudnn*
~~~

# Compiling Cuda Examples
One more thing to do if you want to compile the cuda examples:

~~~bash
$ sudo vim /usr/local/cuda/include/host_config.h
~~~

Comment out line 115:

~~~diff
- #error -- unsupported GNU version! gcc versions later than 4.9 are not supported!
~~~

### Install Tensorflow from Source Code
Most of steps are exactly the same from [Official Guildlines](https://www.tensorflow.org/versions/r0.9/get_started/os_setup.html#installing-from-sources). For the record, I'm using python `2.7.11`.

# Install Bazel
Install Oracle Java first:

~~~bash
$ sudo add-apt-repository ppa:webupd8team/java
$ sudo apt-get update
$ sudo apt-get install oracle-java8-installer
~~~

Other potential dependencies:

~~~bash
$ sudo apt-get install pkg-config zip g++ zlib1g-dev unzip
~~~

Download the installer:

~~~bash
$ wget https://github.com/bazelbuild/bazel/releases/download/0.3.0/bazel-0.3.0-installer-linux-x86_64.sh
~~~

Run the installer

~~~bash
$ chmod a+x bazel-0.2.3-installer-linux-x86_64.sh 
$ ./bazel-0.2.3-installer-linux-x86_64.sh --user
~~~

The `--user` flag install bazel in `$HOME/bin` directory, add enviromental variable to `~/.bashrc`

~~~bash
$ export PATH="$PATH:$HOME/bin"
~~~

# Install other dependencies

~~~bash
# For Python 2.7:
$ sudo apt-get install python-numpy swig python-dev python-wheel
~~~

# Configure the installation

~~~bash
$ ./configure
~~~

~~~
Please specify the location of python. [Default is /usr/bin/python]:
Do you wish to build TensorFlow with Google Cloud Platform support? [y/N] y
Google Cloud Platform support will be enabled for TensorFlow
Do you wish to build TensorFlow with GPU support? [y/N] y
GPU support will be enabled for TensorFlow
Please specify which gcc nvcc should use as the host compiler. [Default is /usr/bin/gcc]:
Please specify the Cuda SDK version you want to use, e.g. 7.0. [Leave empty to use system default]: 7.5
Please specify the location where CUDA 7.5 toolkit is installed. Refer to README.md for more details. [Default is /usr/local/cuda]:
Please specify the Cudnn version you want to use. [Leave empty to use system default]: 5
Please specify the location where cuDNN 5 library is installed. Refer to README.md for more details. [Default is /usr/local/cuda]:
Please specify a list of comma-separated Cuda compute capabilities you want to build with.
You can find the compute capability of your device at: https://developer.nvidia.com/cuda-gpus.
Please note that each additional compute capability significantly increases your build time and binary size.
[Default is: "3.5,5.2"]: 5.2
Setting up Cuda include
Setting up Cuda lib64
Setting up Cuda bin
Setting up Cuda nvvm
Setting up CUPTI include
Setting up CUPTI lib64
Configuration finished
~~~

# Build a Pip Wheel of Tensorflow
You will still build a pip package and install it for python usage:

~~~bash
$ bazel build -c opt //tensorflow/tools/pip_package:build_pip_package
# To build with GPU support:
$ bazel build -c opt --config=cuda //tensorflow/tools/pip_package:build_pip_package
$ bazel-bin/tensorflow/tools/pip_package/build_pip_package /tmp/tensorflow_pkg
# The name of the .whl file will depend on your platform.
$ sudo pip install /tmp/tensorflow_pkg/tensorflow-0.9.0rc0-py2-none-any.whl
~~~

Potentially this process takes a lot of time depending on your machine, ~500 seconds in my case. 

**Important:** Compiler error might occur due to incompatible gcc version. 

Typical errors:

~~~c
error: build fail with cuda: identifier "__builtin_ia32_mwaitx" is undefined
~~~

or

~~~c
error: "memcpy" was not declared in this scope
~~~

I changed the compilation flags of tensorflow in order to make it work [^2] [^3]:

~~~bash
vim third_party/gpus/crosstool/CROSSTOOL
~~~

Add three lines after around line 52:

~~~diff
   cxx_flag: "-std=c++11"
+  cxx_flag: "-D_MWAITXINTRIN_H_INCLUDED"
+  cxx_flag: "-D__STRICT_ANSI__"
+  cxx_flag: "-D_FORCE_INLINES"
   linker_flag: "-lstdc++"
   linker_flag: "-B/usr/bin/"
~~~

For those experiencing mysterious oriental GFW forces, replace one of the git repositories in `tensorflow\workspace.bzl` around line 146:

~~~diff
  native.new_git_repository(
    name = "boringssl_git",
    commit = "e72df93461c6d9d2b5698f10e16d3ab82f5adde3",
-   remote = "https://boringssl.googlesource.com/boringssl",
+   remote = "https://github.com/google/boringssl.git"
    build_file = path_prefix + "boringssl.BUILD",
  )
~~~

# Now Try Tensorflow

~~~
$ cd tensorflow/models/image/mnist
$ python convolutional.py
I tensorflow/stream_executor/dso_loader.cc:108] successfully opened CUDA library libcublas.so.7.5 locally
I tensorflow/stream_executor/dso_loader.cc:108] successfully opened CUDA library libcudnn.so.5 locally
I tensorflow/stream_executor/dso_loader.cc:108] successfully opened CUDA library libcufft.so.7.5 locally
I tensorflow/stream_executor/dso_loader.cc:108] successfully opened CUDA library libcuda.so locally
I tensorflow/stream_executor/dso_loader.cc:108] successfully opened CUDA library libcurand.so.7.5 locally
~~~

---

### References
[^1]: https://www.pugetsystems.com/labs/hpc/NVIDIA-CUDA-with-Ubuntu-16-04-beta-on-a-laptop-if-you-just-cannot-wait-775/
[^2]: https://github.com/fayeshine/tensorflow/commit/6c8c572c12521d706eda692fa7793f90b45dde20
[^3]: https://github.com/tensorflow/tensorflow/issues/1066