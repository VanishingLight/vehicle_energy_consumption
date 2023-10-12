from config import Config
from model.pytorch_models.cnn import CNN
from model.pytorch_models.cvt import ConvolutionalVisionTransformer, QuickGELU, LayerNorm
from model.pytorch_models.rvt import ResidualVisionTransformer, DeepFEC_v2_vehicle_config_model, DeepFEC_v2_vehicle_config_model_2
from __init__ import get_train_test_data
import torch
from torch import nn
from torchsummary import summary
import numpy as np
from functools import partial
from model.metrics import rmse, mape, mae, get_model_save_path
import tensorwatch as tw
import os

# torch.cuda.is_available() checks and returns a Boolean True if a GPU is available, else it'll return False
# is_cuda = torch.cuda.is_available()

# # If we have a GPU available, we'll set our device to GPU. We'll use this device variable later in our code.
# if is_cuda:
#     device = torch.device("cuda")
#     print("GPU is available")
# else:
#     device = torch.device("cpu")
#     print("GPU not available, CPU used")

# Set manual seed for reproducability
torch.manual_seed(50)

conf = Config("config_fig.yaml")
print(conf.observe_length)

# PREPARE DATA
data, arm_shape, train_xs, train_ys, train_arms, train_xp, train_xt, train_xe,\
    train_vehicle_type, train_engine_config, train_gen_weight,\
    test_xs, test_ys, test_arms, test_xp, test_xt, test_xe,\
    test_vehicle_type, test_engine_config, test_gen_weight = \
    get_train_test_data(conf, need_road_network_structure_matrix=True)  # \
print('************** Train - Predict **********')
print('train_xs:', train_xs.shape,  'test_xs:', test_xs.shape, 'train_xp:', train_xp.shape, 'test_xp:',
      test_xp.shape, 'test_xe:', test_xe.shape, 'train_ys:', train_ys.shape, 'test_ys:', test_ys.shape)
print(train_vehicle_type.shape, train_engine_config.shape, train_gen_weight.shape)

if conf.use_vehicle_info:
    train_xs = np.concatenate((train_xs, train_vehicle_type, train_engine_config, train_gen_weight), axis = 3)
    test_xs = np.concatenate((test_xs, test_vehicle_type, test_engine_config, test_gen_weight), axis = 3)

## USE HOLIDAY INFO (train_xe)
# Repeat the last part along axis=1 to match a shape of (23, 955, 22)
train_xe = np.tile(train_xe[:, -1:], (1, 955, 1))
test_xe = np.tile(test_xe[:, -1:], (1, 955, 1))

# print(train_xs[:,:,:,3:4])
# exit()
## MODEL
# Instantiate the model with hyperparameters
# model = CNN()
# print(conf.cvt_spec['NUM_STAGES'])
model = None
if conf.model_name == 'CVT':
    model = ConvolutionalVisionTransformer(
        in_chans=1,
        num_classes=1,
        act_layer=QuickGELU,
        norm_layer=partial(LayerNorm, eps=1e-5),
        spec=conf.cvt_spec
    )
else:
    print("using DeepFEC_v2_vehicle_config_model")
    model = DeepFEC_v2_vehicle_config_model_2(
        in_chans=1,
        num_classes=1,
        act_layer=QuickGELU,
        norm_layer=partial(LayerNorm, eps=1e-5),
        spec=conf.cvt_spec
    )

# We'll also set the model to the device that we defined earlier (default is CPU)
# model = model.to(device)

# Define hyperparameters
n_epochs = conf.epochs
lr=0.01

# Define Loss, Optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

# summary(model, (12, 5), 955)
# exit()

# exit()
# print('== model_stats by tensorwatch ==')
# df = tw.model_stats(
#     model,
#     (23, 955, 12, 5)
# )
# df.to_html(os.path.join('pytorch_model_summary', 'model_summary.html'))
# df.to_csv(os.path.join('pytorch_model_summary', 'model_summary.csv'))
# msg = '*'*20 + ' Model summary ' + '*'*20
# print(
#     '\n{msg}\n{summary}\n{msg}'.format(
#         msg=msg, summary=df.iloc[-1]
#     )
# )
# print('== model_stats by tensorwatch ==')
# exit()
# print("reached here 2")

## TRAIN
for epoch in range(n_epochs + 1):  # loop over the dataset multiple times
    for i in range(train_xs.shape[0]):
        # get the inputs; data is a list of [inputs, labels]
        inputs = torch.from_numpy(train_xs[i]).float()
        holiday_inputs = torch.from_numpy(train_xe[i]).float()
        values = torch.from_numpy(train_ys[i]).float()
        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        # print("input :" , inputs.shape)
        outputs = model(inputs, holiday_inputs)

        loss = criterion(outputs, values)
        loss.backward()
        optimizer.step()

        # print statistics
        print('Epoch: {}/{}.............'.format(epoch, n_epochs), end=' ')
        print("Loss: {:.8f}".format(loss.item()))

print('Finished Training, starting testing')

## Test and calculate accuracy
np_values = np.array([])
np_predicted = np.array([])
with torch.no_grad():
    for j in range(test_xs.shape[0]):
        t_inputs = torch.from_numpy(test_xs[j]).float()
        h_inputs = torch.from_numpy(test_xe[j]).float()
        t_values = torch.from_numpy(test_ys[j]).float()
        predicted = model(t_inputs, h_inputs)

        np_values = np.append(np_values, t_values.numpy())
        np_predicted = np.append(np_predicted, predicted.numpy())

predict = data.min_max_scala.inverse_transform(np_predicted)
y_true = data.min_max_scala.inverse_transform(np_values)
# predict = np_predicted
# y_true = np_values

v_rmse = rmse(predict, y_true)
v_mae = mae(predict, y_true)
v_mape = mape(predict, y_true)

print("RMSE:", v_rmse)
print("MAE:", v_mae)
print("MAPE:", v_mape)
# print(np_values.shape, np_predicted.shape)

# with open('predicted.csv', 'w') as outfile:
#     for data_slice in predict:
#         outfile.write('# New slice\n')
#         np.savetxt(outfile, data_slice)
    
# with open('real.csv', 'w') as outfile:
#     for data_slice in y_true:
#         outfile.write('# New slice\n')
#         np.savetxt(outfile, data_slice)