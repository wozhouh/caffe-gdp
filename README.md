# Caffe-GDP
Caffe-GDP is a branch of [Caffe](https://github.com/BVLC/caffe), which adds a few lines of code in order to enable a global and dynamic filter pruning (GDP) on convolution layers of typical CNN architecture, as described in a newly accepted paper at IJCAI18, [Accelerating Convolutional Networks via Global & Dynamic Filter Pruning](https://www.ijcai.org/proceedings/2018/0336.pdf). The paper mentioned is based on TensorFlow originally. 

The following will introduce how GDP is implemented based on the original Caffe framework, then a guidance to perform GDP on a CNN. If you do not care about details, feel free to skip the first part.

## Inplementation

New members added to the data structure are listed as below.

|**New members** | |
|---| :---: |
|Blob| |
|vector<Dtype*> filter_contribution_2D_; |the channel-wise contribution of filters at the convolution layer |
|vector<Dtype> filter_contrib_; |the contribution of filters at the convolution layer |
|vector<int> filter_mask_; |the mask of filters at the convolution layer |
|**Net** | |
|vector<int> conv_layer_ids_; |the IDs of convolution layers in the net |
|int num_filter_total_; | the total numbers of filters in the net |
|vector<Dtype> filter_contrib_total_; | the collection of filter-wise contribution in the net |
|**BaseConvolutionLayer** | |
|shared_ptr<Blob<Dtype> > masked_weight_; |the masked weight blob which takes part in forward and backward |
|**Solver** | |
|is_pruning etc| added super-parameters at caffe.proto|
 
The GDP iterations is different that it updates the mask according to the ranking of all filters right after back-prop and maked the weight blob before forward of the next iteration.

## instruction

Here are the newly added super-parameters at caffe.proto.

|**super-parameters** | meaning| default|
|---| :---: | :---: |
|is_pruning |whether to perform GDP |false |
|pruning_rate |the remaining proportion of filters |1.0 |
|mask_updating_step | steps of mask updating interval| 1|
|mask_updating_stepsize | how often is the interval changed| 1000|
|mi_policy | the pattern of how the interval is changed ("exp"/"minus") | "minus"|
|log_type | the pattern of how the mask is printed ("debug"/"release") | "debug"|
|log_name | the pattern of how the log of printed mask is named| "mask.log"|

The following is a guideline to perform GDP to a typical CNN, taking LeNet5 as an example.

1 Firstly enter the root directory of Caffe-GDP and train the net from scratch as usual, 

`./build/tools/caffe train -solver examples/mnist/lenet_solver.prototxt`

2 Then turn on the GDP at [prototxt](https://github.com/wozhouh/caffe-gdp/blob/master/examples/mnist/lenet_solver_pruning.prototxt) and set the necessary parameters

`./build/tools/caffe train -solver examples/mnist/lenet_solver_pruning.prototxt -weights examples/mnist/lenet_iter_10000.caffemodel`
 
3 Run a Python [script](https://github.com/wozhouh/caffe-gdp/blob/master/python/caffemodel_channel_pruning.py) to cut the caffemodel aotomatically according to mask.log

`python ./python/auto_caffemodel_pruning.py`

4 (Optional)Fine-Tune
`./build/tools/caffe train -solver examples/mnist/lenet_solver_finetune.prototxt -weights examples/mnist/lenet_iter_3000_pruned.caffemodel`

Now when GDP is finished, we get a caffemodel of about 799kB (pruning_rate: 0.5), which is only 47.4% of the original 1684kB with an accuracy of 98.91% compared to 99.02%. 

GDP is an automatic pruning method of typical CNN architect, which make is thinner and faster.

# Caffe

[![Build Status](https://travis-ci.org/BVLC/caffe.svg?branch=master)](https://travis-ci.org/BVLC/caffe)
[![License](https://img.shields.io/badge/license-BSD-blue.svg)](LICENSE)

Caffe is a deep learning framework made with expression, speed, and modularity in mind.
It is developed by Berkeley AI Research ([BAIR](http://bair.berkeley.edu))/The Berkeley Vision and Learning Center (BVLC) and community contributors.

Check out the [project site](http://caffe.berkeleyvision.org) for all the details like

- [DIY Deep Learning for Vision with Caffe](https://docs.google.com/presentation/d/1UeKXVgRvvxg9OUdh_UiC5G71UMscNPlvArsWER41PsU/edit#slide=id.p)
- [Tutorial Documentation](http://caffe.berkeleyvision.org/tutorial/)
- [BAIR reference models](http://caffe.berkeleyvision.org/model_zoo.html) and the [community model zoo](https://github.com/BVLC/caffe/wiki/Model-Zoo)
- [Installation instructions](http://caffe.berkeleyvision.org/installation.html)

and step-by-step examples.

## Custom distributions

 - [Intel Caffe](https://github.com/BVLC/caffe/tree/intel) (Optimized for CPU and support for multi-node), in particular Xeon processors (HSW, BDW, SKX, Xeon Phi).
- [OpenCL Caffe](https://github.com/BVLC/caffe/tree/opencl) e.g. for AMD or Intel devices.
- [Windows Caffe](https://github.com/BVLC/caffe/tree/windows)

## Community

[![Join the chat at https://gitter.im/BVLC/caffe](https://badges.gitter.im/Join%20Chat.svg)](https://gitter.im/BVLC/caffe?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge)

Please join the [caffe-users group](https://groups.google.com/forum/#!forum/caffe-users) or [gitter chat](https://gitter.im/BVLC/caffe) to ask questions and talk about methods and models.
Framework development discussions and thorough bug reports are collected on [Issues](https://github.com/BVLC/caffe/issues).

Happy brewing!

## License and Citation

Caffe is released under the [BSD 2-Clause license](https://github.com/BVLC/caffe/blob/master/LICENSE).
The BAIR/BVLC reference models are released for unrestricted use.

Please cite Caffe in your publications if it helps your research:

    @article{jia2014caffe,
      Author = {Jia, Yangqing and Shelhamer, Evan and Donahue, Jeff and Karayev, Sergey and Long, Jonathan and Girshick, Ross and Guadarrama, Sergio and Darrell, Trevor},
      Journal = {arXiv preprint arXiv:1408.5093},
      Title = {Caffe: Convolutional Architecture for Fast Feature Embedding},
      Year = {2014}
    }
