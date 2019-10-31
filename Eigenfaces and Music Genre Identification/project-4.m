% Project 4 - PCA
%% Part 1 - Yale Faces B
%% Data Extraction

clear all; close all; clc

directionPath = './data/yalefaces_cropped/CroppedYale/';
direction = dir(directionPath);

faces = [];
averageFaces = [];

tic
for i = 3 : length(direction)
    currentDirection = direction(i).name;
    fileList = dir(strcat(directionPath, currentDirection));
    subfaces = zeros(1,32256);
    sizeSubfaces = 0;
    for j = 3 : length(fileList)
        currentFilePath = strcat(strcat(directionPath, currentDirection), ...
            '/',fileList(j).name);
        face = double(imread(currentFilePath));
        faceA= reshape(face, [1, size(face, 1)*size(face, 2)]);
        faces = [faces; faceA];
        subfaces = subfaces+faceA;
        sizeSubfaces = sizeSubfaces + 1;
    end
    averageFaces = [averageFaces; subfaces/sizeSubfaces];
%     imshow(uint8(reshape(averageFaces(i-2,:), [size(face, 1), size(face, 2)])));
end
save("averageFaces.mat", "averageFaces", "-mat");
toc

%%
clc
[m, n] = size(faces);
mn = mean(faces, 2);
faces2 = faces - repmat(mn, 1, n);
[U, S, V] = svd(faces2'/sqrt(n-1), 'econ');
plot([1:size(S)],S/sum(S));

%%
for j=1:3
    ff=S(:,1:j)*V(1:j,1:j)*D(:,1:j)'; % modal projections
%     subplot(2,2,j+1)
%     surfl(X,T,ff), shading interp, colormap(gray)
%     set(gca,'Zlim',[0.5 2])
end

%% Co-variance matrix
C = cov(faces2');

%% SVD of co-variance matrix
[U,S,V] = svd(C);
face1 = reshape(U(:,2),size(face, 1), size(face, 2));
pcolor(flipud(face1)), shading INTERP, colormap(gray)
lambda = diag(S);

%% Plot Eigenfaces
for i = 1:9
    subplot(3,3,i)
    face = reshape(U(:,i),size(face, 1), size(face, 2));
    pcolor(flipud(face)), shading INTERP, colormap(gray)
end
figure()
semilogy(lambda(1:9), 'ko', 'Linewidth', [2])

%% Projection of Average Faces
for i = 1:size(averageFaces, 1)
    projFace = U'*averageFaces(i, :)';
    subplot(7, 6, i)
    bar(projFace(1:20)), set(gca, 'Xlim', [0 20], 'Ylim', [-4000 4000]), ...
        text(14, -3000, sprintf('Face %d', i));
end

%% Plot Average Faces
for i = 1:size(averageFaces, 1)
    subplot(7, 6, i)
    imshow(uint8(reshape(averageFaces(i, :), [size(face, 1), size(face, 2)])));
end

%% Music classification
%% Test 1 - Band classification
clear all; close all; clc

mainDirectionPath = "music-samples/test-1/sample-60/";
mainDirection = dir(mainDirectionPath);

% songs = zeros(4, 160000, 2);
songs = [];
classification = [];
averageSongs = [];
tic
for i = 3 : length(mainDirection)
    currentArtist = mainDirection(i).name;
    currentArtistDirection = dir(strcat(mainDirectionPath, currentArtist));
    numberOfSongs = length(currentArtistDirection) - 2;
    averageSong = zeros(1, 160000);
    for j = 3 : length(currentArtistDirection)
        currentSongPath = strcat(strcat(mainDirectionPath, currentArtist), ...
            '/',currentArtistDirection(j).name);
        [song,Fs] = audioread(currentSongPath);
        for p = 1:1
            randomStartPoint = randi([0 length(song) - 160000],1);
            randomSample = song(randomStartPoint:randomStartPoint+160000-1,1);
            songs = [songs; randomSample'];
            classification = [classification; string(currentArtist)];
        end
%         for p = 1:1
%             randomStartPoint = randi([0 length(song) - 160000],1,1);
%             randomSample = song(randomStartPoint:randomStartPoint+160000-1);
%             songs = [songs; randomSample'];
%         end
        averageSong = averageSong + randomSample';
    end
    averageSongs = [averageSongs; averageSong/numberOfSongs];
end
toc

%% Play all samples
for i = 1:size(songs)
    playblocking(audioplayer(songs(i,:),Fs));
end

%% Fourier Transform

songsF = [];
for i = 1:size(songs, 1)
    songsF = [songsF; fftshift(fft(songs(i,:)))];
end

% plot(abs(fftshift(songsF(2,:))))

%% SVD

[u,s,v] = svd(abs(songsF)', 'econ');

figure()
plot(diag(s), 'ko', 'Linewidth', [2])

figure()
for i = 1:4
    subplot(2,2,i)
    plot((u(:,i)))
%     plot(u(:,i))
end

figure()
plot3(v(1:10,2), v(1:10,3), v(1:10,4),'ko', 'Linewidth', [2])
hold on
plot3(v(11:20,2), v(11:20,3), v(11:20,4),'ro', 'Linewidth', [2])
hold on
plot3(v(21:end,2), v(21:end,3), v(21:end,4),'yo', 'Linewidth', [2])

%% Training

n1 = 10;
n2 = 10;

q1 = randperm(n1);
q2 = randperm(n2);

xDay = v(1:n1, 2:3);
xGong = v(n1+1:end, 2:3);
xLone = v()

% xTrain = [xDay(q1(1:n1/2), :); xGong(q2(1:n2/2), :)];
% xTest = [xDay(q1(n1/2+1:end), :); xGong(q2(n2/2+1:end), :)];

xTrain = [xDay; xGong];
xTest = xTrain;

classification2 = zeros(30, 1);
classification2(1:10) = ones(10, 1);
classification2(11:20) = 2*ones(10, 1);

% figure()
% plot3(xTrain(1:8,1), xTrain(1:8,2), v(1:8,3), 'ko', 'Linewidth', [2])
% hold on
% plot3(xTrain(9:16,1), xTrain(9:16,2), v(9:16,3), 'ro', 'Linewidth', [2])
% figure()
% plot3(xTest(1:4,1), xTest(1:4,2), v(1:4,3), 'ko', 'Linewidth', [2])
% hold on
% plot3(xTest(5:8,1), xTest(5:8,2), v(5:8,3), 'ro', 'Linewidth', [2])

svm = fitcsvm(xTrain, classification, 'KernelFunction', 'rbf');
Mdl = fitcknn(xTrain,classification,'NumNeighbors',5,'Standardize',1);
res = predict(svm, xTest);
res2 = predict(Mdl, xTest);


%%

n1 = 10;
n2 = 10;
n3 = 10;

c = cvpartition(10,'Kfold',5);
err = zeros(c.NumTestSets,1);
for i = 1 : c.NumTestSets
    trIdx = c.training(i);
    teIdx = c.test(i);

    n = c.NumObservations;

    xDay = v(1:n, 2:5);
    xGong = v(n+1:2*n, 2:5);
    xLone = v(2*n+1:end, 2:5);

    classification2 = [classification(1:8);classification(13:20);classification(23:30)];
    classification2Test = [classification(9:12);classification(21:22)];

    xTrain = [xDay(trIdx, :); xGong(trIdx, :); xLone(trIdx, :)];
    xTest = [xDay(teIdx, :); xGong(teIdx, :); xLone(teIdx, :)];
    svm = fitcknn(xTrain,classification2,'NumNeighbors',5,'Standardize',1);
    res = predict(svm, xTest);
    err(i) = sum(~strcmp(res,classification2Test));
end
cvErr = sum(err)/(i*sum(c.TestSize))
