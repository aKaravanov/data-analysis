% HW3 - PCA
%% Test 1 - Initilization 

clear all; close all; clc

disp("Getting Camera 1")
camOneRgb = load('cam1_1'); 
camOneRgb = camOneRgb.("vidFrames1_1");

disp("Getting Camera 2")
camTwoRgb = load('cam2_1'); 
camTwoRgb = camTwoRgb.("vidFrames2_1");

disp("Getting Camera 3")
camThreeRgb = load('cam3_1'); 
camThreeRgb = camThreeRgb.("vidFrames3_1");

% Test 1 - Setting-Up

close all; clc

timeSize = min([size(camOneRgb, 4) size(camTwoRgb, 4) size(camThreeRgb, 4)]);

xSize = min([size(camOneRgb, 1) size(camTwoRgb, 1) size(camThreeRgb, 1)]);

ySize = min([size(camOneRgb, 2) size(camTwoRgb, 2) size(camThreeRgb, 2)]);

camOneGrayMat = zeros(xSize, ySize, timeSize);
camTwoGrayMat = camOneGrayMat;
camThreeGrayMat = camOneGrayMat;

camOneGrayA = zeros(timeSize, xSize*ySize);
camTwoGrayA = camOneGrayA;
camThreeGrayA = camOneGrayA;

for i = 1:timeSize
    camOneGrayMat(:,:,i) = imresize(double(rgb2gray(camOneRgb(:,:,:,i))), ...
        [xSize,ySize]);
    camTwoGrayMat(:,:,i) = imresize(double(rgb2gray(camTwoRgb(:,:,:,i))), ...
        [xSize,ySize]);
    camThreeGrayMat(:,:,i) = imresize(double(rgb2gray(camThreeRgb(:,:,:,i))), ...
        [xSize,ySize]);
    camOneGrayA(i,:) = reshape(camOneGrayMat(:,:,i), [1, xSize*ySize]);
    camTwoGrayA(i,:) = reshape(camTwoGrayMat(:,:,i), [1, xSize*ySize]);
    camThreeGrayA(i,:) = reshape(camThreeGrayMat(:,:,i), [1, xSize*ySize]);
end

allCamsGrayA = [camOneGrayA; camTwoGrayA; camThreeGrayA];

allCamsGrayMat = zeros(xSize, ySize, timeSize, 3);
allCamsGrayMat(:,:,:,1) = camOneGrayMat;
allCamsGrayMat(:,:,:,2) = camTwoGrayMat;
allCamsGrayMat(:,:,:,3) = camThreeGrayMat;

leftBorders = 4*[70; 60; 60];
rightBorders = 4*[100; 90; 131];

[xCoordinates, yCoodinates] = findCoordinates(allCamsGrayMat, leftBorders, ...
    rightBorders, timeSize, xSize, ySize);

% Test 1 - PCA

%Creating X vector
xVec = zeros(6, timeSize);
for i = 1:6
    if rem(i, 2) == 0
        xVec(i, :) = yCoodinates(i/2, :);
    else 
        xVec(i, :) = xCoordinates(rem(i,2),:);
    end
end

[m,n] = size(xVec);   %  compute data size
mn = mean(xVec,2); %  compute mean for each row
xVec = xVec - repmat(mn,1,n);  % subtract mean
covarianceMat = cov(xVec'); % compute covariance

[V,D]=eig(covarianceMat);      % eigenvectors(V)/eigenvalues(D)
lambda=diag(D);    % get eigenvalue

[dummy,m_arrange]=sort(-1*lambda);  % sort in decreasingorder
lambda=lambda(m_arrange);
V=V(:,m_arrange);

Y = V'*xVec;

% Test 1 - Output

subplot(1,2,1)
plot(lambda, 'ko', 'Linewidth', [1.5])
title('Singular Values')
xlabel('Number of singular value')
ylabel('Value of singular value')
subplot(1,2,2)
plot([1:timeSize], Y(1,:), "r", [1:timeSize], Y(2,:), "b", [1:timeSize], ...
    Y(3,:), "g", [1:timeSize], Y(4,:), "y", "Linewidth", [2])
title('Projections using PCA')
xlabel('Time Count, Frames')
ylabel('Displacement, Number of Pixels')

%% Test 2 - Initilization 

clear all; close all; clc

disp("Getting Camera 1")
camOneRgb = load('cam1_2'); 
camOneRgb = camOneRgb.("vidFrames1_2");

disp("Getting Camera 2")
camTwoRgb = load('cam2_2'); 
camTwoRgb = camTwoRgb.("vidFrames2_2");

disp("Getting Camera 3")
camThreeRgb = load('cam3_2'); 
camThreeRgb = camThreeRgb.("vidFrames3_2");

% Test 2 - Setting-Up

close all; clc

timeSize = min([size(camOneRgb, 4) size(camTwoRgb, 4) size(camThreeRgb, 4)]);

xSize = min([size(camOneRgb, 1) size(camTwoRgb, 1) size(camThreeRgb, 1)]);

ySize = min([size(camOneRgb, 2) size(camTwoRgb, 2) size(camThreeRgb, 2)]);

camOneGrayMat = zeros(xSize, ySize, timeSize);
camTwoGrayMat = camOneGrayMat;
camThreeGrayMat = camOneGrayMat;

camOneGrayA = zeros(timeSize, xSize*ySize);
camTwoGrayA = camOneGrayA;
camThreeGrayA = camOneGrayA;

for i = 1:timeSize
    camOneGrayMat(:,:,i) = imresize(double(rgb2gray(camOneRgb(:,:,:,i))), ...
        [xSize,ySize]);
    camTwoGrayMat(:,:,i) = imresize(double(rgb2gray(camTwoRgb(:,:,:,i))), ...
        [xSize,ySize]);
    camThreeGrayMat(:,:,i) = imresize(double(rgb2gray(camThreeRgb(:,:,:,i))), ...
        [xSize,ySize]);
    camOneGrayA(i,:) = reshape(camOneGrayMat(:,:,i), [1, xSize*ySize]);
    camTwoGrayA(i,:) = reshape(camTwoGrayMat(:,:,i), [1, xSize*ySize]);
    camThreeGrayA(i,:) = reshape(camThreeGrayMat(:,:,i), [1, xSize*ySize]);
end

allCamsGrayA = [camOneGrayA; camTwoGrayA; camThreeGrayA];

allCamsGrayMat = zeros(xSize, ySize, timeSize, 3);
allCamsGrayMat(:,:,:,1) = camOneGrayMat;
allCamsGrayMat(:,:,:,2) = camTwoGrayMat;
allCamsGrayMat(:,:,:,3) = camThreeGrayMat;

leftBorders = 4*[70; 50; 60];
rightBorders = 4*[100; 100; 131];

[xCoordinates, yCoodinates] = findCoordinates(allCamsGrayMat, leftBorders, ...
    rightBorders, timeSize, xSize, ySize);

% Test 2 - PCA

%Creating X vector
xVec = zeros(6, timeSize);
for i = 1:5
    if rem(i, 2) == 0
        xVec(i, :) = yCoodinates(i/2, :);
    else 
        xVec(i, :) = xCoordinates(rem(i,2),:);
    end
end

[m,n] = size(xVec);   %  compute data size
mn = mean(xVec,2); %  compute mean for each row
xVec = xVec - repmat(mn,1,n);  % subtract mean
covarianceMat = cov(xVec'); % compute covariance

[V,D]=eig(covarianceMat);      % eigenvectors(V)/eigenvalues(D)
lambda=diag(D);    % get eigenvalue

[dummy,m_arrange]=sort(-1*lambda);  % sort in decreasingorder
lambda=lambda(m_arrange);
V=V(:,m_arrange);

Y = V'*xVec;

% Test 2 - Output

subplot(1,2,1)
plot(lambda, 'ko', 'Linewidth', [1.5])
title('Singular Values')
xlabel('Number of singular value')
ylabel('Value of singular value')
subplot(1,2,2)
plot([1:timeSize], Y(1,:), "r", [1:timeSize], Y(2,:), "b", [1:timeSize], ...
    Y(3,:), "g", [1:timeSize], Y(4,:), "y", "Linewidth", [2])
title('Projections using PCA')
xlabel('Time Count, Frames')
ylabel('Displacement, Number of Pixels')

%% Test 3 - Initilization 

clear all; close all; clc

disp("Getting Camera 1")
camOneRgb = load('cam1_3'); 
camOneRgb = camOneRgb.("vidFrames1_3");

disp("Getting Camera 2")
camTwoRgb = load('cam2_3'); 
camTwoRgb = camTwoRgb.("vidFrames2_3");

disp("Getting Camera 3")
camThreeRgb = load('cam3_3'); 
camThreeRgb = camThreeRgb.("vidFrames3_3");

% Test 3 - Setting-Up

close all; clc

timeSize = min([size(camOneRgb, 4) size(camTwoRgb, 4) size(camThreeRgb, 4)]);

xSize = min([size(camOneRgb, 1) size(camTwoRgb, 1) size(camThreeRgb, 1)]);

ySize = min([size(camOneRgb, 2) size(camTwoRgb, 2) size(camThreeRgb, 2)]);

camOneGrayMat = zeros(xSize, ySize, timeSize);
camTwoGrayMat = camOneGrayMat;
camThreeGrayMat = camOneGrayMat;

camOneGrayA = zeros(timeSize, xSize*ySize);
camTwoGrayA = camOneGrayA;
camThreeGrayA = camOneGrayA;

for i = 1:timeSize
    camOneGrayMat(:,:,i) = imresize(double(rgb2gray(camOneRgb(:,:,:,i))), ...
        [xSize,ySize]);
    camTwoGrayMat(:,:,i) = imresize(double(rgb2gray(camTwoRgb(:,:,:,i))), ...
        [xSize,ySize]);
    camThreeGrayMat(:,:,i) = imresize(double(rgb2gray(camThreeRgb(:,:,:,i))), ...
        [xSize,ySize]);
    camOneGrayA(i,:) = reshape(camOneGrayMat(:,:,i), [1, xSize*ySize]);
    camTwoGrayA(i,:) = reshape(camTwoGrayMat(:,:,i), [1, xSize*ySize]);
    camThreeGrayA(i,:) = reshape(camThreeGrayMat(:,:,i), [1, xSize*ySize]);
end

allCamsGrayA = [camOneGrayA; camTwoGrayA; camThreeGrayA];

allCamsGrayMat = zeros(xSize, ySize, timeSize, 3);
allCamsGrayMat(:,:,:,1) = camOneGrayMat;
allCamsGrayMat(:,:,:,2) = camTwoGrayMat;
allCamsGrayMat(:,:,:,3) = camThreeGrayMat;

leftBorders = 4*[70; 60; 60];
rightBorders = 4*[100; 90; 131];

[xCoordinates, yCoodinates] = findCoordinates(allCamsGrayMat, leftBorders, ...
    rightBorders, timeSize, xSize, ySize);

% Test 3 - PCA

%Creating X vector
xVec = zeros(6, timeSize);
for i = 1:5
    if rem(i, 2) == 0
        xVec(i, :) = yCoodinates(i/2, :);
    else 
        xVec(i, :) = xCoordinates(rem(i,2),:);
    end
end

[m,n] = size(xVec);   %  compute data size
mn = mean(xVec,2); %  compute mean for each row
xVec = xVec - repmat(mn,1,n);  % subtract mean
covarianceMat = cov(xVec'); % compute covariance

[V,D]=eig(covarianceMat);      % eigenvectors(V)/eigenvalues(D)
lambda=diag(D);    % get eigenvalue

[dummy,m_arrange]=sort(-1*lambda);  % sort in decreasingorder
lambda=lambda(m_arrange);
V=V(:,m_arrange);

Y = V'*xVec;

% Test 3 - Output

subplot(1,2,1)
plot(lambda, 'ko', 'Linewidth', [1.5])
title('Singular Values')
xlabel('Number of singular value')
ylabel('Value of singular value')
subplot(1,2,2)
plot([1:timeSize], Y(1,:), "r", [1:timeSize], Y(2,:), "b", [1:timeSize], ...
    Y(3,:), "g", [1:timeSize], Y(4,:), "y", "Linewidth", [2])
title('Projections using PCA')
xlabel('Time Count, Frames')
ylabel('Displacement, Number of Pixels')

%% Test 4 - Initilization 

clear all; close all; clc

disp("Getting Camera 1")
camOneRgb = load('cam1_4'); 
camOneRgb = camOneRgb.("vidFrames1_4");

disp("Getting Camera 2")
camTwoRgb = load('cam2_4'); 
camTwoRgb = camTwoRgb.("vidFrames2_4");

disp("Getting Camera 3")
camThreeRgb = load('cam3_4'); 
camThreeRgb = camThreeRgb.("vidFrames3_4");

% Test 4 - Setting-Up

close all; clc

timeSize = min([size(camOneRgb, 4) size(camTwoRgb, 4) size(camThreeRgb, 4)]);

xSize = min([size(camOneRgb, 1) size(camTwoRgb, 1) size(camThreeRgb, 1)]);

ySize = min([size(camOneRgb, 2) size(camTwoRgb, 2) size(camThreeRgb, 2)]);

camOneGrayMat = zeros(xSize, ySize, timeSize);
camTwoGrayMat = camOneGrayMat;
camThreeGrayMat = camOneGrayMat;

camOneGrayA = zeros(timeSize, xSize*ySize);
camTwoGrayA = camOneGrayA;
camThreeGrayA = camOneGrayA;

for i = 1:timeSize
    camOneGrayMat(:,:,i) = imresize(double(rgb2gray(camOneRgb(:,:,:,i))), ...
        [xSize,ySize]);
    camTwoGrayMat(:,:,i) = imresize(double(rgb2gray(camTwoRgb(:,:,:,i))), ...
        [xSize,ySize]);
    camThreeGrayMat(:,:,i) = imresize(double(rgb2gray(camThreeRgb(:,:,:,i))), ...
        [xSize,ySize]);
    camOneGrayA(i,:) = reshape(camOneGrayMat(:,:,i), [1, xSize*ySize]);
    camTwoGrayA(i,:) = reshape(camTwoGrayMat(:,:,i), [1, xSize*ySize]);
    camThreeGrayA(i,:) = reshape(camThreeGrayMat(:,:,i), [1, xSize*ySize]);
end

allCamsGrayA = [camOneGrayA; camTwoGrayA; camThreeGrayA];

allCamsGrayMat = zeros(xSize, ySize, timeSize, 3);
allCamsGrayMat(:,:,:,1) = camOneGrayMat;
allCamsGrayMat(:,:,:,2) = camTwoGrayMat;
allCamsGrayMat(:,:,:,3) = camThreeGrayMat;

leftBorders = 4*[70; 60; 60];
rightBorders = 4*[100; 90; 131];

[xCoordinates, yCoodinates] = findCoordinates(allCamsGrayMat, leftBorders, ...
    rightBorders, timeSize, xSize, ySize);

% Test 4 - PCA

%Creating X vector
xVec = zeros(6, timeSize);
for i = 1:5
    if rem(i, 2) == 0
        xVec(i, :) = yCoodinates(i/2, :);
    else 
        xVec(i, :) = xCoordinates(rem(i,2),:);
    end
end

[m,n] = size(xVec);   %  compute data size
mn = mean(xVec,2); %  compute mean for each row
xVec = xVec - repmat(mn,1,n);  % subtract mean
covarianceMat = cov(xVec'); % compute covariance

[V,D]=eig(covarianceMat);      % eigenvectors(V)/eigenvalues(D)
lambda=diag(D);    % get eigenvalue

[dummy,m_arrange]=sort(-1*lambda);  % sort in decreasingorder
lambda=lambda(m_arrange);
V=V(:,m_arrange);

Y = V'*xVec;

% Test 4 - Output

subplot(1,2,1)
plot(lambda, 'ko', 'Linewidth', [1.5])
title('Singular Values')
xlabel('Number of singular value')
ylabel('Value of singular value')
subplot(1,2,2)
plot([1:timeSize], Y(1,:), "r", [1:timeSize], Y(2,:), "b", [1:timeSize], ...
    Y(3,:), "g", [1:timeSize], Y(4,:), "y", "Linewidth", [2])
title('Projections using PCA')
xlabel('Time Count, Frames')
ylabel('Displacement, Number of Pixels')
