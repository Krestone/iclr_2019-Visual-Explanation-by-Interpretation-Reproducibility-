clc;
clear;
% load dataset
imdb = load('imdb.mat');

% load locs.csv
locs = load('locs.csv');

% run preparation
run ../matlab/vl_setupnn

% load W
features = load('selected_features/feautres.csv');
features100 = load('selected_features/features_100.csv');

% train/test data preparation
train_images = imdb.images.data(:,:,1,find(imdb.images.set==1));
test_images = imdb.images.data(:,:,1,find(imdb.images.set==3));
train_labels = imdb.images.labels(1, find(imdb.images.set==1));
test_labels = imdb.images.labels(1, find(imdb.images.set==3));

% load pre-trained model
MNIST_model = load('net-epoch-20.mat');
MNIST_model.net.layers{end}.type = 'softmax';

% ablated model
net_ablated = network_ablation(MNIST_model.net, features100, locs,  false);
net_ablated.layers{end}.type = 'softmax';

% random ablated model
net_ablated_random = network_ablation_random(MNIST_model.net, features100, locs, 175, false);
net_ablated_random.layers{end}.type = 'softmax';









% get output
res = vl_simplenn(MNIST_model.net, train_images);
res_ablated = vl_simplenn(net_ablated, train_images);
res_ablated_random = vl_simplenn(net_ablated_random, train_images);


% get predicted labels
train_labels_pred = int32(zeros(1,60000));
for i_img=1:60000
    vec = squeeze(res(9).x(1,1,:,i_img));
    [maxV,label] = max(vec);
    train_labels_pred(1, i_img) = label;
end


train_ablated_labels_pred = int32(zeros(1,60000));
for i_img=1:60000
    vec = squeeze(res_ablated(9).x(1,1,:,i_img));
    [maxV,label] = max(vec);
    train_ablated_labels_pred(1, i_img) = label;
end

train_ablated_random_labels_pred = int32(zeros(1,60000));
for i_img=1:60000
    vec = squeeze(res_ablated_random(9).x(1,1,:,i_img));
    [maxV,label] = max(vec);
    train_ablated_random_labels_pred(1, i_img) = label;
end




function net_ablated = network_ablation(net, W, locations, convOnly)
[row, col] = find(W);

filters_ablated = unique(row);
[num_filters_ablated, dummy]=size(filters_ablated);
fprintf('#filters ablated %0.0f\n', num_filters_ablated);

net_ablated = net;

for i = 1:num_filters_ablated
   
    layer = locations(filters_ablated(i),1);
    filter = locations(filters_ablated(i),2);
    fprintf('Ablated: Layer %.0f, Filter %.0f \n', layer, filter);
    if(convOnly == false)
        net_ablated.layers{1, layer}.weights{1,1}(:,:,:,filter) = 0;
        net_ablated.layers{1, layer}.weights{1,2}(1, filter) = 0;
    elseif(layer ~= 5)
        net_ablated.layers{1, layer}.weights{1,1}(:,:,:,filter) = 0;
        net_ablated.layers{1, layer}.weights{1,2}(1, filter) = 0;
    end
end
end

function net_ablated_random = network_ablation_random(net, W, locations, num_ablation, convOnly)
[num_total_filter, dummy] = size(W);
filters_ablated = datasample(1:num_total_filter, num_ablation, 'Replace', false);
if(convOnly)
    filters_ablated = datasample(1:70, num_ablation, 'Replace', false);
end
    
fprintf('#filters ablated %0.0f\n', num_ablation);

net_ablated_random = net;

for i = 1:num_ablation
   
    layer = locations(filters_ablated(i),1);
    filter = locations(filters_ablated(i),2);
    fprintf('Ablated: Layer %.0f, Filter %.0f \n', layer, filter);
    net_ablated_random.layers{1, layer}.weights{1,1}(:,:,:,filter) = 0;
    net_ablated_random.layers{1, layer}.weights{1,2}(1, filter) = 0;

        
end
end










    



