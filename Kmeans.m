% Kmeans.m
% Raunak Bhojwani
% Assignment 7
% CS74 - Machine Learning

%load data
clear all;
close all;
iris=load('data/iris.txt'); 
Y=iris(:,end); 
X=iris(:,1:2);

% 1a
plot(X(:,1),X(:,2),'k*','MarkerSize',5);
title('Feature 1 compared to Feature 2');
xlabel('Feature 1');
ylabel('Feature 2');

% 1b for k=2 
[idx,C] = kmeans(X,2);

figure;
plot(X(idx==1,1),X(idx==1,2),'r.','MarkerSize',12)
hold on
plot(X(idx==2,1),X(idx==2,2),'b.','MarkerSize',12)
plot(C(:,1),C(:,2),'kx',...
     'MarkerSize',15,'LineWidth',3)
legend('Cluster 1','Cluster 2','Centroids',...
       'Location','NW')
title 'Cluster Assignments and Centroids for K=2'
hold off

% 1b for k=5
[idx,C] = kmeans(X,5);

figure;
plot(X(idx==1,1),X(idx==1,2),'r.','MarkerSize',12)
hold on
plot(X(idx==2,1),X(idx==2,2),'g.','MarkerSize',12)
plot(X(idx==3,1),X(idx==3,2),'b.','MarkerSize',12)
plot(X(idx==4,1),X(idx==4,2),'c.','MarkerSize',12)
plot(X(idx==5,1),X(idx==5,2),'m.','MarkerSize',12)
plot(C(:,1),C(:,2),'kx',...
     'MarkerSize',15,'LineWidth',3)
legend('Cluster 1','Cluster 2', 'Cluster 3','Cluster 4','Cluster 5', 'Centroids',...
       'Location','NW')
title 'Cluster Assignments and Centroids for K=5'
xlabel('Feature 1');
ylabel('Feature 2');
hold off

%1c
n=10;
IDX = zeros(148, n);
C = zeros(n*5,2);
SUMD = zeros(n, 1);

for i = 1:n
    [tempIDX,tempC, tempSUMD] = kmeans(X,5);
    SUMD(i,1) = sum(tempSUMD);
    IDX(:,i) = tempIDX;
    C(1 + (i-1)*5:5 + (i-1)*5,:) = tempC;
end

[waste,minIDX] = min(SUMD);
 
x1 = min(X(:,1))-.5:0.01:max(X(:,1)+.5);
x2 = min(X(:,2))-.5:0.01:max(X(:,2)+.5);
[newx1,newx2] = meshgrid(x1,x2);
XMeshed = [newx1(:),newx2(:)];
idxRegions = kmeans(XMeshed,5,'MaxIter',1,'Start',C(1 + (minIDX-1)*5:5 + (minIDX-1)*5,:));

figure;
gscatter(XMeshed(:,1),XMeshed(:,2),idxRegions,...
            [0,.75,0.75;0.75,0,0.75;0.75,0.75,0;0,.5,0;.5,.5,0],'..'); 
hold on;
plot(X(:,1),X(:,2),'k*','MarkerSize',5);
plot(C(1 + (minIDX-1)*5:5 + (minIDX-1)*5,1),C(1 + (minIDX-1)*5:5 + (minIDX-1)*5,2),'kx','MarkerSize',15,'MarkerFaceColor','b','LineWidth',3)
xlabel 'Feature 1';
ylabel 'Feature 2';
title 'Cluster Assignments and Centroids for repeated K=5';
legend('Cluster 1','Cluster 2','Cluster 3', 'Cluster 4', 'Cluster 5','Data','Centroids','Location','NW');
hold off;
