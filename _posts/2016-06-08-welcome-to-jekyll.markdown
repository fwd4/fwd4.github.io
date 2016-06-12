---
layout: post
title:  "Installation of Tensorflow"
date:   2016-06-08 14:29:52 +0800
categories: Miscellany
---

What I need:
- Tensorflow with GPU support
- Compile from source code for potential hacking into the code

What I have:
- Ubuntu 16.04
- GCC 5.3.1
- Xeon E5 1620
- Nvidia Geforce Titan X

What I achieved:
- Tensorflow 0.9rc from source code
- GPU support enabled
- CUDA 7.5
- cuDNN v5

Tensorflow is now only compatible with Linux and MacOS right now. For some reason, Windows servers are pervasive in my case, so the installation process has deviated from the right path from the beginning. I've struggled for quite some time to try to find a way to install tensorflow in a linux virtual machine environment. But for all the hypervisor softwares that I tried, VirtualBox, VMware Player and Hyper-V respectively, the GPU is invisible to the virtual machine. There are hypervisors in Windows like VMware vSphere supporting GPU passthrough to virtual machine though, which I cannot afford. So do not avoid the trouble of installing a dual-boot system, as you will get more trouble in linux virtual machines. Of course you can still dip your toes with tensorflow in VMs if you don't want GPU support.  



You’ll find this post in your `_posts` directory. Go ahead and edit it and re-build the site to see your changes. You can rebuild the site in many different ways, but the most common way is to run `jekyll serve`, which launches a web server and auto-regenerates your site when a file is updated.

To add new posts, simply add a file in the `_posts` directory that follows the convention `YYYY-MM-DD-name-of-post.ext` and includes the necessary front matter. Take a look at the source for this post to get an idea about how it works.

Jekyll also offers powerful support for code snippets:

{% highlight ruby %}
def print_hi(name)
  puts "Hi, #{name}"
end
print_hi('Tom')
#=> prints 'Hi, Tom' to STDOUT.
{% endhighlight %}

Check out the [Jekyll docs][jekyll-docs] for more info on how to get the most out of Jekyll. File all bugs/feature requests at [Jekyll’s GitHub repo][jekyll-gh]. If you have questions, you can ask them on [Jekyll Talk][jekyll-talk].

[jekyll-docs]: http://jekyllrb.com/docs/home
[jekyll-gh]:   https://github.com/jekyll/jekyll
[jekyll-talk]: https://talk.jekyllrb.com/
