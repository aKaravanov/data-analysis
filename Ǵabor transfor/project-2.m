% HW1 - Gabor transforms
%-----------------------------%
% Part 1

clear all; close all; clc
load handel

v = y'/2;

n = length(v);
L = n/Fs;
vf = v(1:end - 1);
k=(2*pi/L)*[0:n/2-1 -n/2:-1]; ks=fftshift(k);

t2 = (1:length(v))/Fs;
t = t2(1:n-1);

width = [5, 20, 300];
tslide_one = 0:1:t(end);
tslide_two = 0:0.1:t(end);
tslide_three = 0:0.05:t(end);

Vst_spec_one = [];
Vst_spec_two = [];
Vst_spec_three = [];
tslide = tslide_three;
for j=1:length(tslide)
    sOne = (abs(t - tslide(j)) <= 1/width(1)); % Shannon
    sTwo = (abs(t - tslide(j)) <= 1/width(2)); % Shannon
    sThree = (abs(t - tslide(j)) <= 1/width(3)); % Shannon
    VsOne = sOne.*vf; VstOne = fft(VsOne);
    VsTwo = sTwo.*vf; VstTwo = fft(VsTwo);
    VsThree = sThree.*vf; VstThree = fft(VsThree);
    Vst_spec_one = [Vst_spec_one; abs(fftshift(VstOne))];
    Vst_spec_two = [Vst_spec_two; abs(fftshift(VstTwo))];
    Vst_spec_three = [Vst_spec_three; abs(fftshift(VstThree))];
end

Vst_spec_two_slide_one = [];
tslide = tslide_one;
for j=1:length(tslide)
    s = (abs(t - tslide(j)) <= 1/width(1)); % Shannon
    Vs = s.*vf; Vst = fft(Vs);
    Vst_spec_two_slide_one = [Vst_spec_two_slide_one; abs(fftshift(Vst))];
end

Vst_spec_two_slide_three = [];
tslide = tslide_three;
for j=1:length(tslide)
    s = (abs(t - tslide(j)) <= 1/width(3)); % Shannon
    Vs = s.*vf; Vgt = fft(Vs);
    Vst_spec_two_slide_three = [Vst_spec_two_slide_three; abs(fftshift(Vst))];
end

tslide = tslide_two;
Vgt_spec_two = [];
Vst_spec_two_slide_two = [];
Vmt_spec_two = [];
for j=1:length(tslide)
    g = exp(-width(2)*(t - tslide(j)).^2); % Gaussian
    s = (abs(t - tslide(j)) <= 1/width(2)); % Shannon
    m = 2.*(1 - ((t-tslide(j))/width(2).^-1).^2)...
        .*exp(-((t-tslide(j)).^2)/...
        (2.*width(2).^-2))/(sqrt(3.*width(2).^-1)...
        .*pi^(1/4)); % Ricker
    Vg = g.*vf; Vgt = fft(Vg);
    Vs = s.*vf; Vst = fft(Vs);
    Vm = m.*vf; Vmt = fft(Vm);
    if (j == length(tslide)/2)
        figure
        subplot(2,3,1:3)
        plot(t,vf,'k',t,g,'r',t,s,'g',t,m,'b')
        axis([4 4.8 -3 5])
        xlabel('Time, sec');
        ylabel('Amplitude');
        title('Original Signal and Filters');
        subplot(2,3,4)
        plot(t,Vg)
        axis([4 4.8 -0.5 0.5])
        xlabel('Time, sec');
        ylabel('Amplitude');
        title('Gaussian Filtered signal');
        subplot(2,3,5)
        plot(t,Vs)
        axis([4 4.8 -0.5 0.5])
        xlabel('Time, sec');
        ylabel('Amplitude');
        title('Shannon Filtered signal');
        subplot(2,3,6)
        plot(t,Vm)
        axis([4 4.8 -0.5 0.5])
        xlabel('Time, sec');
        ylabel('Amplitude');
        title('Mexican Filtered signal');
    end
    Vgt_spec_two = [Vgt_spec_two; abs(fftshift(Vgt))];
    Vst_spec_two_slide_two = [Vst_spec_two_slide_two; abs(fftshift(Vst))];
    Vmt_spec_two = [Vmt_spec_two; abs(fftshift(Vmt))];
end
%%
close all;
figure
subplot(3,1,1)
pcolor(tslide_three,ks,Vst_spec_one.'), ...
    shading interp, colormap(hot)
str = sprintf('Shannon Spectrogram with width parameter 5');
title(str)
xlabel('Time, Sec')
ylabel('Frequency')

subplot(3,1,2)
pcolor(tslide_three,ks,Vst_spec_two.'), shading interp,...
    colormap(hot)
str = sprintf('Shannon Spectrogram with width parameter 20');
title(str)
xlabel('Time, Sec')
ylabel('Frequency')

subplot(3,1,3)
pcolor(tslide_three,ks,Vst_spec_three.'), shading interp,...
    colormap(hot)
str = sprintf('Shannon Spectrogram with width parameter 200');
title(str)
xlabel('Time, Sec')
ylabel('Frequency')

figure
subplot(3,1,1)
pcolor(tslide_one,ks,Vst_spec_two_slide_one.'), ...
    shading interp, colormap(hot)
str = sprintf('Undersampled Shannon Spectrogram');
title(str)
xlabel('Time, Sec')
ylabel('Frequency')

subplot(3,1,2)
pcolor(tslide_two,ks,Vst_spec_two_slide_two.'), shading interp,...
    colormap(hot)
str = sprintf('Normally Sampled Shannon Spectrogram');
title(str)
xlabel('Time, Sec')
ylabel('Frequency')

subplot(3,1,3)
pcolor(tslide_three,ks,Vst_spec_two.'), shading interp,...
    colormap(hot)
str = sprintf('Oversampled Shannon Spectrogram');
title(str)
xlabel('Time, Sec')
ylabel('Frequency')

figure 
subplot(3,1,1)
pcolor(tslide_two,ks,Vgt_spec_two.'), shading interp,...
    colormap(hot)
str = sprintf('Gaussian spectogram');
title(str)
xlabel('Time, Sec')
ylabel('Frequency')
subplot(3,1,2)
pcolor(tslide_two,ks,Vst_spec_two_slide_two.'), shading interp,...
    colormap(hot)
str = sprintf('Shannon spectogram');
title(str)
xlabel('Time, Sec')
ylabel('Frequency')
subplot(3,1,3)
pcolor(tslide_two,ks,Vmt_spec_two.'), shading interp,...
    colormap(hot)
str = sprintf('Mexican spectogram');
title(str)
xlabel('Time, Sec')
ylabel('Frequency')

%% Part 2

clear all; close all; clc

L=16; % record time in seconds
y=audioread('music1.wav'); 
Fs=length(y)/L;
v = y'/2;
t = (1:length(v))/Fs;
n=length(v);
k=(2*pi/L)*[0:n/2-1 -n/2:-1]; 
ks=fftshift(k);

width = [10000000];
tslide_one = 0.2:0.53:14;

for i = 1:length(width)
    figure(2*i - 1)
    title('Piano recording')
    
    Sgt_spec = [];
    frequencies = [];
    for j=1:length(tslide_one)
        g = exp(-width(i)*(t - tslide_one(j)).^10);
        Sg = g.*v; %filtered with gaussian
        Sgt = fft(Sg); %fft gaussian
        [val, index] = max(abs(fftshift(Sgt)));
        frequency = ks(index)/(2*pi);
        frequencies = [frequencies, frequency];
        Sgt_spec = [Sgt_spec; abs(fftshift(Sgt))];
    end
    frequencies = abs(frequencies);
    
    figure
    scatter(tslide_one(1:length(tslide_one)),frequencies(1:length(tslide_one)));
    xlabel('Time, Sec');
    ylabel('Frequency, Hz');
    title('Frequencies of notes played (Piano)');
    axis([0 L 200 400])
    drawnow   
end
%%
clear all; close all; clc

L=14; % record time in seconds
y=audioread('music2.wav'); 
Fs=length(y)/L;
v = y'/2;
t = (1:length(v))/Fs;
n=length(v);
k=(2*pi/L)*[0:n/2-1 -n/2:-1]; 
ks=fftshift(k);

width = [10000000];
tslide = 0:0.44:13;

for i = 1:length(width)
    figure(2*i - 1)
    title('Recording')
    
    Sgt_spec = [];
    frequencies = [];
    for j=1:length(tslide)
        g = exp(-width(i)*(t - tslide(j)).^10);
        Sg = g.*v; %filtered with gaussian
        Sgt = fft(Sg); %fft gaussian
        [val, index] = max(abs(fftshift(Sgt)));
        frequency = ks(index)/(2*pi);
        frequencies = [frequencies, frequency];
        Sgt_spec = [Sgt_spec; abs(fftshift(Sgt))];
    end
    frequencies = abs(frequencies);
    
    figure
    scatter(tslide(1:length(tslide)),frequencies(1:length(tslide)));
    xlabel('Time, Sec');
    ylabel('Frequency, Hz');
    title('Frequencies of notes played (Recording)');
    axis([0 L 700 1200])
    drawnow   
end



