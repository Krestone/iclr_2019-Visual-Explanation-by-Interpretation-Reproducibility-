clear;
clc;
load('net-epoch-20.mat')
load('imdb.mat')
addpath('../matlab/simplenn/')
addpath('../matlab/')
run ../matlab/vl_setupnn


net.layers{end}.type = 'softmax';




for i = 1:numel(net.layers)
    layerType{i}=net.layers{1,i}.type;
end

train_images = images.data(:,:,1,find(images.set==1));
train_images_ = train_images; 
res= vl_simplenn(net, train_images_);



% layer 1 processing
for i_img = 1:60000
    for flr = 1:20
        l1(flr,i_img) = norm(res(2).x(:,:,flr,i_img), 2);
    end
end

for i_img = 1:60000
    normalization = norm(l1(:,i_img), 1);
    l1(:,i_img)=l1(:,i_img)/normalization;
end

% layer 3 processing
for i_img = 1:60000
    for flr = 1:50
        l3(flr,i_img) = norm(res(4).x(:,:,flr,i_img), 2);
    end
end

for i_img = 1:60000
    normalization = norm(l3(:,i_img), 1);
    l3(:,i_img)=l3(:,i_img)/normalization;
end

% layer 6 processing
for i_img = 1:60000
    for flr = 1:500
        l6(flr,i_img) = norm(res(7).x(:,:,flr,i_img), 2);
    end
end

for i_img = 1:60000
    normalization = norm(l6(:,i_img), 1);
    l6(:,i_img)=l6(:,i_img)/normalization;
end

X = [l1;l3;l6];

locs = int32(zeros(570,2));
locs(1:20,1)=1;
locs(21:70,1)=3;
locs(71:end,1)=5;

locs(1:20,2)=1:20;
locs(21:70,2)=1:50;
locs(71:end,2)=1:500;

L=int32(zeros(10,60000));

for i_img=1:60000
    vec = squeeze(res(9).x(1,1,:,i_img));
    [maxV,label] = max(vec);
    label;
    L(label,i_img)=1;
end

csvwrite('L.csv', L)
csvwrite('X.csv', X)
csvwrite('locs.csv', locs)




