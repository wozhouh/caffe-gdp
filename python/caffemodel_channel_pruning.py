# -*- coding: utf-8 -*-

import sys
import caffe

def read_mask_log(mask_log):
    mask = []
    f_mask = open(mask_log, "r") 
    lines = f_mask.readlines()
    for line in lines:
        mask_layer = []
        cnt = 0
        for s in line:
            if s == ' ':
                continue
            if s == '1':
                mask_layer.append(cnt)
            cnt += 1
        mask.append(mask_layer)
    f_mask.close()
    return mask

def update_masked_prototxt(prototxt_in, prototxt_out):
    f_r = open(prototxt_in, "r")
    lines = f_r.readlines()
    with open(prototxt_out, "w") as f_w:
        layer_found = False
        all_found = False
        idx = 0
        # find the pruned layer in prototxt
        for line in lines:
            if not (all_found or layer_found) and conv_layers[idx] in line:
                layer_found = True
            if layer_found and ("num_output: " in line):
                num = str(len(mask[idx]))
                f_w.write("    num_output: "+num+"\n")    
                layer_found = False
                idx += 1 
                all_found = idx == len(mask)
            else:
                f_w.write(line)
    f_r.close()
    f_w.close()

if __name__ == "__main__":
    
    # import pycaffe
    caffe_root = '/home/processyuan/caffe/caffe-gdp/'
    python_dir = caffe_root+'python/'
    sys.path.append(python_dir)
   
    # name of files
    caffe.set_mode_cpu()
    workspace = caffe_root+'examples/mnist/'
    train_prototxt_in = workspace+'lenet_train_test.prototxt'
    deploy_prototxt_in = workspace+'lenet.prototxt'
    train_prototxt_out = train_prototxt_in[0:-9] + '_pruned.prototxt'
    deploy_prototxt_out = deploy_prototxt_in[0:-9] + '_pruned.prototxt'
    caffemodel_in = workspace+'lenet_iter_5000.caffemodel'
    caffemodel_out = caffemodel_in[0:-11]+ '_pruned.caffemodel'
    mask_log = workspace+'mask.log'
  
    # import the model
    model = caffe.proto.caffe_pb2.NetParameter()
    f_caffemodel = open(caffemodel_in, 'rb')
    model.ParseFromString(f_caffemodel.read())
    f_caffemodel.close()
    
    # select the convolution layers and save their names in conv_layers, corresponding successive layers in succ_layer
    conv_layers = [] # list of convolution layers whose order is the same as that in prototxt
    succ_layers = {} # dict whose keys are layers which come successively after a convolution layer, and values are the corresponding convolution layers
    layers = model.layer
    layer_pruned = ""
    for layer in layers:
        if layer_pruned != "" and (layer.type == "Convolution" or layer.type == "InnerProduct"):
            succ_layers[layer.name] = layer_pruned
            layer_pruned = ""
        if layer.type == "Convolution":
            conv_layers.append(layer.name)
            layer_pruned = layer.name
    
    # import mask.log as list: order of convolution layers as they appear in prototxt
    mask = read_mask_log(mask_log)
    assert len(mask) == len(conv_layers)

    # write a new prototxt for the pruned caffemodel
    update_masked_prototxt(train_prototxt_in, train_prototxt_out)
    update_masked_prototxt(deploy_prototxt_in, deploy_prototxt_out)
    net_in = caffe.Net(deploy_prototxt_in, caffemodel_in, caffe.TEST)
    net_out = caffe.Net(deploy_prototxt_out, caffe.TEST)
    
    # generate the new caffemodel
    weight_masked = {} # dict whose keys are layer names of which the caffemodel weights have to change, values are changed weights
    bias_masked = {} # same as above
    for layer in net_in.params.keys():
        if layer in conv_layers:
            filter_sel = mask[conv_layers.index(layer)]
            weight_old = net_in.params[layer][0].data
            weight_masked[layer] = weight_old[filter_sel, :, :, :]
            # suppose every convolution layer has a bias 
            bias_old = net_in.params[layer][1].data
            bias_masked[layer] = bias_old[filter_sel]
            
    for layer in net_in.params.keys():
        if layer in succ_layers.keys():
            succ_layer = succ_layers[layer]
            channel_sel = mask[conv_layers.index(succ_layer)]
            # layers that both itself and its succesive layer are pruned (convolution layers)
            if layer in conv_layers:
                weight_masked[layer] = (weight_masked[layer])[:, channel_sel, :, :]
            # layers whose succesive layer are pruned, but itself not pruned (inner-product layers)
            else:
                bias_masked[layer] = net_in.params[layer][1].data
                weight_old = net_in.params[layer][0].data
                input_dim_featuremap = (weight_old.shape[1]) / (net_in.params[succ_layer][0].data.shape[0])
                input_dim_sel = []
                for m in channel_sel:
                    for k in range(input_dim_featuremap):
                        input_dim_sel.append(input_dim_featuremap*m+k)                       
                weight_masked[layer] = weight_old[:, input_dim_sel]
                    
    for layer in net_in.params.keys(): 
        if layer in weight_masked.keys():
            net_out.params[layer][0].data[...] = weight_masked[layer]
            net_out.params[layer][1].data[...] = bias_masked[layer]
        else:
            weight = net_in.params[layer]
            for k, w in enumerate(weight):
                net_out.params[layer][k].data[...] = w.data

    # save the new caffemodel
    net_out.save(caffemodel_out)
