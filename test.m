directory = pwd; 
% Load and explore image data
imds = imageDatastore(directory,'IncludeSubfolders',true,'FileExtensions','.jpg', 'LabelSource', 'foldernames');

inputSize = [227 227];
labelCount = countEachLabel(imds);
[imdsTrain,imdsValidation] = splitEachLabel(shuffle(imds),.8, 'randomized');

% Image Complement
% while hasdata(imdsTrain)
%     [img, info] = read(imdsTrain);
%     img = imcomplement(img);
%     Write to the location 
%     imwrite(img,info.Filename);
% end
% imdsTrain = imageDatastore(directory,'IncludeSubfolders',true,'FileExtensions','.jpg', 'LabelSource', 'foldernames');

% Image Rotation, Reflection, and Reflection 
all_Augmenter = imageDataAugmenter('RandRotation',[-60,60],'RandXReflection', true, 'RandXTranslation',[-60 60]);
rotationAugmenter = imageDataAugmenter('RandRotation',[-20,20]);
reflectionXAugmenter = imageDataAugmenter('RandXReflection', true);
reflectionYAugmenter = imageDataAugmenter('RandYReflection', true);
reflectionXYAugmenter = imageDataAugmenter('RandXReflection', true, 'RandYReflection', true);
translationXAugmenter = imageDataAugmenter('RandXTranslation',[-3 3]);
translationYAugmenter = imageDataAugmenter('RandYTranslation',[-3 3]);
translationXYAugmenter = imageDataAugmenter('RandXTranslation',[-3 3],'RandYTranslation',[-3 3]);

% augimdsTrain = augmentedImageDatastore(inputSize(1:2),imdsTrain,'ColorPreprocessing','rgb2gray');
augimdsTrain = augmentedImageDatastore(inputSize(1:2),imdsTrain,'ColorPreprocessing','gray2rgb');
augimdsValidation = augmentedImageDatastore(inputSize(1:2),imdsValidation, 'ColorPreprocessing','gray2rgb');

% Define network architecture
layers = [
    % Specify the image size & channel size (grayscale or RGB)
    % 1 = grayscale 3 = RGB 
    % An image input layer inputs images to a network and applies data normalization.
    %     imageInputLayer([227 227 3],'DataAugmentation','randcrop')
    imageInputLayer([227 227 3])
    
    % filterSize = 3 = height & width of filters the training function uses
    % while scanning images
    % numFilters = 8 = # of neurons that connect to the same region of the
    % input (Determines feature maps)
    % Use the 'Padding' name-value pair to add padding to the input feature 
    % map. For a convolutional layer with a default stride of 1, 'same' 
    % padding ensures that the spatial output size is the same as the input size. 
    convolution2dLayer(3,8,'Padding','same')
    
    % Batch normalization layers normalize the activations and gradients 
    % propagating through a network, making network training an easier 
    % optimization problem. Use batch normalization layers between convolutional 
    % layers and nonlinearities, such as ReLU layers, to speed up network 
    % training and reduce the sensitivity to network initialization.
    batchNormalizationLayer
    % Activation Function rectified linear unit (ReLU
    reluLayer
    
    % down-sampling operation that reduces the spatial size of the feature 
    % map and removes redundant spatial information. Down-sampling makes it
    % possible to increase the number of filters in deeper convolutional 
    % layers without increasing the required amount of computation per layer.
    maxPooling2dLayer(2,'Stride',2)
    
    convolution2dLayer(3,16,'Padding','same')
    batchNormalizationLayer
    reluLayer
    
    maxPooling2dLayer(2,'Stride',2)
    
    convolution2dLayer(3,32,'Padding','same')
    batchNormalizationLayer
    reluLayer
    
    % a fully connected layer is a layer in which the neurons connect to
    % all the neurons in the preceding layer.
    % This layer combines all the features learned by the previous layers 
    % across the image to identify the larger patterns. The last fully 
    % connected layer combines the features to classify the images.
    fullyConnectedLayer(2)
    % The softmax activation function normalizes the output of the fully 
    % connected layer. The output of the softmax layer consists of positive 
    % numbers that sum to one, which can then be used as classification
    % probabilities by the classification layer. 
    softmaxLayer
    %  This layer uses the probabilities returned by the softmax activation
    % function for each input to assign the input to one of the mutually 
    % exclusive classes and compute the loss.
    classificationLayer];

% Specify Training Options
% An epoch is a full training cycle on the entire training data set.
options = trainingOptions('sgdm', ...
    'InitialLearnRate',0.01, ...
    'MaxEpochs',3, ...
    'Shuffle','every-epoch', ...
    'ValidationData',augimdsValidation, ...
    'ValidationFrequency',30, ...
    'Verbose',false);
% Train network
net = trainNetwork(shuffle(augimdsTrain),layers,options);

% Training 
YPred_Train = classify(net,augimdsTrain);
YTrain = imdsTrain.Labels;
training_accuracy = sum(YPred_Train == YTrain)/numel(YTrain)
cat_training_accuracy = sum(YPred_Train == 'Cat')/numel(YTrain)
dog_training_accuracy = sum(YPred_Train == 'Dog')/numel(YTrain)

% Validation
YPred_Validation = classify(net,augimdsValidation);
YValidation = imdsValidation.Labels;
validation_accuracy = sum(YPred_Validation == YValidation)/numel(YValidation)
cat_validation_accuracy = sum(YPred_Validation == 'Cat')/numel(YValidation)
dog_validation_accuracy = sum(YPred_Validation == 'Dog')/numel(YValidation)

% Reset images to normal

% while hasdata(imdsTrain)
%     [img, info] = read(imdsTrain);
%     img = imcomplement(img);
%     % Write to the location 
%     imwrite(img,info.Filename);
% end
% imdsTrain = imageDatastore(directory,'IncludeSubfolders',true,'FileExtensions','.jpg', 'LabelSource', 'foldernames');
