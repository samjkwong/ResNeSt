#!/usr/bin/env python3

import numpy as np
import torch
import benchmark
from resnet import ResNet, Bottleneck


def get_model(model_name):
    models = {
      'resnest50': ResNet(
                      Bottleneck, [3, 4, 6, 3],
                      radix=2, groups=1, bottleneck_width=64,
                      deep_stem=True, stem_width=32, avg_down=True,
                      avd=True, avd_first=False),
      'resnest101': ResNet(
                      Bottleneck, [3, 4, 23, 3],
                      radix=2, groups=1, bottleneck_width=64,
                      deep_stem=True, stem_width=64, avg_down=True,
                      avd=True, avd_first=False),
      'resnest200': ResNet(
                      Bottleneck, [3, 24, 36, 3],
                      radix=2, groups=1, bottleneck_width=64,
                      deep_stem=True, stem_width=64, avg_down=True,
                      avd=True, avd_first=False),
      'resnest269': ResNet(
                      Bottleneck, [3, 30, 48, 8],
                      radix=2, groups=1, bottleneck_width=64,
                      deep_stem=True, stem_width=64, avg_down=True,
                      avd=True, avd_first=False),
    }

    return models[model_name]

def time_fwd():
    model_name = 'resnest269'
    model = get_model(model_name)
    model = model.cuda(device=torch.cuda.current_device())
    im_size, batch_size = 416, 16
    time = benchmark.compute_time_eval(model, im_size, batch_size)
    print('-'*50)
    print('Latency (ms): {}'.format(time*1000))
    print('Throughput: {}'.format(batch_size / time))


if __name__ == '__main__':
    time_fwd()

