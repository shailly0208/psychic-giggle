clc;
clear all;
close all;
image=imread('L5.jpeg');
%figure(1); subplot(2,2,1);
%imshow(image); title('orig rgb');

bw = rgb2gray(image);   %RGB to gray
%figure(1); subplot(2,2,2);
%imshow(bw); title('grayscale');

threshold = 80;                   % custom Gray threshold value
A_bw = bw > threshold;
%figure(1); subplot(2,2,3);
%imshow(A_bw); title('threshold');

% BW= imbinarize(bw);                 %to binary
% figure(1); subplot(2,2,4);
% imshow(BW);title('binarized');
% bw = rgb2gray(image);     %RGB to grayscale 
% figure(1); subplot(2,2,2);
% imshow(bw);title('grayscale');

image1=im2double(A_bw);          %grayscale to double
figure(1); subplot(1,2,1);
imshow(image1); title('double');

%[c,r] = imfindcircles(BW,[500,800]);
% T = adaptthresh(image, 0.9);            %Adaptive Thresholding
% BW2=imbinarize(image,(rgb2gray(T)));                %Segmentating using the adaptive threshold
% imshow(BW2);title('Adaptive Thresholding');

%%shndsdjaskdashdjkdshjk
% b1=zeros(256);
% %figure(2); subplot(2,2,1);
% %imshow(b1); title('orig black bg');
% a1=imresize(image1,[398,604]);  	% rescaling the images to facilitate subtraction
% b1=imresize(b1,[398,604]);
% %figure(2); subplot(2,2,2);
% %imshow(a1); title('resized orig');
% %figure(2); subplot(2,2,3);
% %imshow(b1);title('resized black bg');

% c=imsubtract(a1,b1); 		% subtracts each element in b2 from the corresponding element in a2 
% i=imadjust(c); 
% figure(3); subplot(1,2,1);
% imshow(c); title('final non-adjusted');
% figure(3);subplot(1,2,2);
% imshow(i);title('final adjusted');
% linkaxes;

% se = strel('square',2);              %creating a structuring element
% background = imerode(a1,se);          %Obtaining the background
% figure(4);subplot(2,2,1);
% imshow(background);title('Background image');
% 
% se1 = strel('rectangle',[2,2]);              %creating a structuring element
% background1 = imerode(a1,se1);          %Obtaining the background
% figure(4);subplot(2,2,2);
% imshow(background1);title('Background image');
% 
% se2 = strel('cube',2);              %creating a structuring element
% background2 = imerode(a1,se2);          %Obtaining the background
% figure(4);subplot(2,2,3);
% imshow(background2);title('Background image');
% 
% se3 = strel('sphere',2);              %creating a structuring element
% background3 = imerode(a1,se3);          %Obtaining the background
% figure(4);subplot(2,2,4);
% imshow(background3);title('Background image');
% linkaxes;
% stats=regionprops('table',a1,'Centroid','MajorAxisLength','MinorAxisLength');
% centres=stats.Centroid;
% diameters=mean([stats.MajorAxisLength stats.MinorAxisLength],2);
% radii=diameters/2;
% hold on;
% viscircles(centres,radii);
% hold off;

[centers,radii] = imfindcircles(A_bw,[30 8000]);                    %[50 8000]        FINDING CIRCLES' RADII & CENTRES
centersStrong5 = centers(1:7); 
radiiStrong5 = radii(1:3);

edges_sobel=edge(A_bw, 'sobel');                                    % APPLYING SOBEL FILTER
figure(1); subplot(1,2,2);
imshow(edges_sobel); title('Sobel filter');

% circ=[centers,radii];
% k=viscircles(circ,centers,radii);
% figure(3);
% imshow(k);
 
%for =1:8;
%     imageSizeX=8;
%     imageSizeY=8;
%     [columns, rows]=meshgrid(1:imageSizeX, 1:imageSizeY);
%     centerX=centers(:,1);
%     centerY=centers(:,2);
%     circlePixels=((rows-centerY).^2)+((columns-centerX).^2)<=(radii.^2);
%end
%elementsIncircle = edges_sobel(circlePixels);

%viscircles(centers, radii,'EdgeColor','r','LineStyle','-');         %PLOTTING CIRCLES W/ Radii's & Centre's; prints on the previous plot
for k = 1:8
    % Clear the axes.
    cla
    % Fix the axis limits.
    xlim([0 1000])
    ylim([0 1000])
    % Set the axis aspect ratio to 1:1.
    axis fill
    % Set a title.
    title('axes')
    % Display the circles.
    viscircles(centers,radii);
    % Pause for 1 second.
    pause(1)
end
mask = viscircles(centers, radii,'EdgeColor','b','LineStyle',':');    %size(A_bw)    
mask(1:417,1:849) = true;                                     %(correct: 1:417,1:849) 200,200
% figure(2);
% imshow(mask);
Inew = edges_sobel.*mask;
figure(3);
imshow(Inew); title('masked?');
%h=visboundaries(mask,'Color','b');
% bw = activecontour(A_bw,mask,200,'edge');                  % A_bw, mask, 200          
% k=visboundaries(bw,'Color','y');
% figure(3);title('Blue - Initial Contour, Red - Final Contour');

m1=medfilt2(edges_sobel,[1 1]);    %bw
%figure(4);subplot(2,2,1); 
%imshow(m1);title('Median filter of size 1x1'); 
% B=[0 1 0; 1 1 1; 0 1 0];                           
% open_ = imopen(m1,B);                           
% figure(4); subplot(2,2,2);
% imshow(open_); title('opened');                       %displaying the open image
% dil = imdilate(m1,B);                             %dilating the image; returns the dilated image
% figure(4); subplot(2,2,3);
% imshow(dil); title('dilated');       

cellStats = regionprops(m1,'all');
cellAreas = [cellStats(:).Area];
hold on
[m1 p]=bwlabel(m1);
prop=regionprops(m1,'BoundingBox');
figure(4);
img = imshow(m1);
sum = size(prop,1);
hold on
for n=1:sum
rectangle('Position',prop(n).BoundingBox,'EdgeColor','c','LineWidth',8)
end
hold off
set(img, 'AlphaData',0.9);   
title(sprintf('Dirt patch/es detected: %i ',sum));