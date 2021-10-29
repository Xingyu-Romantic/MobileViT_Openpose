from paddle import nn


def get_parameters(model, predicate, lr):
    print(model)
    print(model.children())
    for module in model.children():
        for param_name, param in module.named_parameters():
            print('*' * 10, 'start')
            print(param_name)
            print('*' * 10, 'mid')
            print(module)
            print('*' *10, 'end')
            if predicate(module, param_name):
                print(module, param_name)
                break
                #module.create_parameter()


def get_parameters_conv(model, name):
    return get_parameters(model, lambda m, p: isinstance(m, nn.Conv2D) and m.groups == 1 and p == name)


def get_parameters_conv_depthwise(model, name):
    return get_parameters(model, lambda m, p: isinstance(m, nn.Conv2D)
                                              and m.groups == m.in_channels
                                              and m.in_channels == m.out_channels
                                              and p == name)


def get_parameters_bn(model, name):
    return get_parameters(model, lambda m, p: isinstance(m, nn.BatchNorm2D) and p == name)