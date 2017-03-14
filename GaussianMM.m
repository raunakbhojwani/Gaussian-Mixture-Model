% GaussianMM.m
% Raunak Bhojwani
% Assignment 7
% CS74 - Machine Learning

%load data
clear all
close all
iris=load('data/iris.txt'); 
Y=iris(:,end); 
X=iris(:,1:2);

%2a for K=2
plot(X(:,1),X(:,2),'k*','MarkerSize',5);

options = statset('Display','final');
gm = fitgmdist(X,2,'Options',options);
gmPDF = @(X,Y)pdf(gm,[X Y]);

hold on
h = ezcontour(gmPDF,[4 9],[1 5]);
title('Scatter Plot and PDF Contour for K=2')
xlabel 'Feature 1';
ylabel 'Feature 2';
hold off

%2a for K=5
figure;
plot(X(:,1),X(:,2),'k*','MarkerSize',5);

options = statset('Display','final');
gm = fitgmdist(X,5,'Options',options);
gmPDF = @(X,Y)pdf(gm,[X Y]);

hold on
h = ezcontour(gmPDF,[4 9],[1 5]);
title('Scatter Plot and PDF Contour for K=5')
xlabel 'Feature 1';
ylabel 'Feature 2';
hold off

%2b
figure;
plot(X(:,1),X(:,2),'k*','MarkerSize',5);
hold on;

options = statset('Display','final');
n = 10;
bestLogLikelihood = 1000;
bestGaussianMM = 0;

for i = 1:n
    gm = fitgmdist(X,5,'Options',options);
    if  bestLogLikelihood > gm.NegativeLogLikelihood
        bestLogLikelihood = gm.NegativeLogLikelihood;
        bestGaussianMM = gm;
    end
end

disp(bestLogLikelihood);

gmPDF = @(X,Y)pdf(bestGaussianMM,[X Y]);
h = ezcontour(gmPDF,[4 9],[1 5]);
title('Scatter Plot and PDF Contour for repeated K=5')
xlabel('Feature 1');
ylabel('Feature 2');
hold off

    
