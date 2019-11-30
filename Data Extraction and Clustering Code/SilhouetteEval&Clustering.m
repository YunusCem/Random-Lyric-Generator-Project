Dataset = out1{:,:};   
save('Dataset.txt', 'Dataset', '-ASCII','-append');
FeatureVector = normalize(Dataset,'range');
E = evalclusters(FeatureVector,'kmeans','silhouette','klist',[1:10])%Silhouette Evaluation to determine the optinal k value
K = E.OptimalK; %No. of Segments
figure;
plot(E)
title(sprintf('Optimal K value =%.4f',K), 'FontSize', 18);

[kIDs, kC] = kmeans(FeatureVector, K, 'Display', 'iter', 'MaxIter', 200, 'Start', 'sample');
DataMean = mean(Dataset);
DataStd = std(Dataset);
save('ClusterMeans.txt', 'kC', '-ASCII','-append');
save('ClusterIndex.txt', 'kIDs', '-ASCII','-append');
save('DataMean.txt', 'DataMean', '-ASCII','-append');
save('DataStd.txt', 'DataStd', '-ASCII','-append');
DataMin = min(Dataset);
save('DataMin.txt', 'DataMin', '-ASCII','-append');
DataMax = max(Dataset);
save('DataMax.txt', 'DataMax', '-ASCII','-append');
