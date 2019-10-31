%% HW5 - Background Subtraction in Video Streams
clear all; close all; clc

disp("Getting Film")
video = VideoReader('mov7.mp4');

disp("Processing It")
v = read(video);
disp("Done")
% save('mov1.mat', 'v', '-mat')

%% Processing Video
clc

timeSize = size(v, 4);

xSize = round(size(v, 1)/4);

ySize = round(size(v, 2)/4);

videoGrayScale = zeros(timeSize, xSize*ySize);

dt = 1; 
t = 0:dt:timeSize;

disp("Working On Video Matrix")
for i = 1:timeSize
    videoGrayScale(i, :) = reshape(imresize(double(rgb2gray(v(:,:,:,i))), [xSize, ySize]), 1, xSize*ySize);
end
disp("Done Working On Video Matrix")

%%  Preparing Data for SVD
clc

disp("Preparing Matrix X1")
X1 = videoGrayScale(1:end-1, :)';
disp("Preparing Matrix X2")
X2 = videoGrayScale(2:end, :)';
disp("Done Preparing Matricies")

%% SVD
clc

tic
disp("Preforming SVD")
[U2,Sigma2,V2] = svd(X1, 'econ');
disp("Done with SVD")
toc

%% Finding Number of Modes Needed
clc

figure()
subplot(1,2,1)
plot(diag(Sigma2(1:40, 1:40))/sum(diag(Sigma2)), 'o');
xlabel("Singular Value Index")
ylabel("Variance of Given Singular Value")
title("Importance of Each Singular Value")
subplot(1,2,2)
to_show = U2(:,1);
to_show = reshape(to_show,[xSize, ySize]);
pcolor(flipud(abs(to_show))), shading interp, colormap(gray);
title('Corresponding 1st mode')
axis off


%% Post-SVD
clc

% Rank Reduction 
r=1; 
U=U2(:,1:r); 
Sigma=Sigma2(1:r,1:r); 
V=V2(:,1:r);

% A Tilde and DMD Modes
Atilde = U'*X2*V/Sigma;
[W,D] = eig(Atilde);
Phi=X2*V/Sigma*W;

% DMD Spectrum
mu=diag(D);
omega=log(mu)/dt;

omegToUse = min(abs(omega)); % Smallest Mode

disp("Done with post SVD")

%% Plot Omegas

figure()
plot(omega,'ko','Linewidth',[2]), grid on, axis([-2 2 -1 1]), set(gca,'Fontsize',[14])
title({'Eigenvalues of $\tilde{A}$'},'Interpreter','latex')

%% The DMD Solution
clc 

b = Phi\X1(:,1);
time_dynamics = zeros(r,timeSize);
for iter = 1:timeSize
    time_dynamics(:,iter) = (b.*exp(omega*t(iter)));
end
X_dmd = Phi*time_dynamics;

disp("Done with finding DMD solution")

%% Finding sparse with residual
clc

X_sparse = videoGrayScale' - abs(X_dmd);
disp("Done with finding sparse with residual")

%% Finding residual
clc

Residual = X_sparse .* (X_sparse < 0);
disp("Done finding residual")

%% Removing residual
clc

X_dmd = abs(X_dmd);
X_sparse = abs(X_sparse);
disp("Done removing residual")

%% Comparison plot
clc

figure()
subplot(3,3,1)
to_show = videoGrayScale(1,:); 
to_show = reshape(to_show,[xSize, ySize]);
pcolor(flipud(abs(to_show))), shading interp, colormap(gray);
title('Original Video at t = 0')
axis off
subplot(3,3,2)
to_show = X_dmd(:,1); 
to_show = reshape(to_show,[xSize, ySize]);
pcolor(flipud(abs(to_show))), shading interp, colormap(gray);
title('Background at t = 0')
axis off
subplot(3,3,3)
to_show = X_sparse(:,1); 
to_show = reshape(to_show,[xSize, ySize]);
pcolor(flipud(abs(to_show))), shading interp, colormap(gray);
title('Foreground at t = 0')
axis off
subplot(3,3,4)
to_show = videoGrayScale(round(timeSize/2),:);
to_show = reshape(to_show,[xSize, ySize]);
pcolor(flipud(abs(to_show))), shading interp, colormap(gray);
title('Original Video at t = T/2')
axis off
subplot(3,3,5)
to_show = X_dmd(:,round(timeSize/2)); 
to_show = reshape(to_show,[xSize, ySize]);
pcolor(flipud(abs(to_show))), shading interp, colormap(gray);
title('Background at t = T/2')
axis off
subplot(3,3,6)
to_show = X_sparse(:,round(timeSize/2)); 
to_show = reshape(to_show,[xSize, ySize]);
pcolor(flipud(abs(to_show))), shading interp, colormap(gray);
title('Foreground at t = T/2')
axis off
subplot(3,3,7)
to_show = videoGrayScale(timeSize,:);
to_show = reshape(to_show,[xSize, ySize]);
pcolor(flipud(abs(to_show))), shading interp, colormap(gray);
title('Original Video at t = T')
axis off
subplot(3,3,8)
to_show = X_dmd(:,timeSize); 
to_show = reshape(to_show,[xSize, ySize]);
pcolor(flipud(abs(to_show))), shading interp, colormap(gray);
title('Background at t = T')
axis off
subplot(3,3,9)
to_show = X_sparse(:,timeSize); 
to_show = reshape(to_show,[xSize, ySize]);
pcolor(flipud(abs(to_show))), shading interp, colormap(gray);
title('Foreground at t = T')
axis off