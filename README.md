# Master's thesis "Feature Selection in Hyperspectral Images for Classification of Apple Proliferation Disease"

## Abstract 

Hyperspectral images provide abundant information by having hundreds of spectral bands in the ranges of VNIR and SWIR wavelengths which are not visible to the naked eye but can reveal interesting information about the objects. The provided information can expose chemical features of objects and thus hyperspectral images become popular among researcher and is used in a wide range of domains like medical, biotechnology, food, and agriculture. But due to a wide range of contiguous spectral bands in hyperspectral images, there is the problem of dealing with such a large amount of data volume effectively and efficiently?

The aim of this thesis is twofold, firstly to find the suitability of hyperspectral images usage for the classification of Apple proliferation disease. Secondly, it provides a comparative study of different feature selection approaches t-test, information gain, SFS, SBS, SFFS, lasso penalized logistic regression, and random forest embedded feature ranking. 

The study is based on three machine learning algorithms, logistic regression, support vector machines, and random forest which all of them can equally well classify apple proliferation disease with 97% accuracy using hyperspectral images however the size of selected feature subsets vary from each other.

Our experiments show that the use of random forest in the wrapper methods results in relatively compact feature subsets compared to logistic regression and SVM. Moreover, using RF wrapped by SBS resulted in the most compact feature subset having only 7 features selected from 186 features. Based on our experiments SBS outperformed SFS and its floating variant SFFS from the point of compactness of feature subset maintaining high performance.  

Information gain filter method, provided fairly small feature subsets of 19 features out of 186 features using random forest, however subsets resulted from logistic regression and SVM are relatively larger and if being forced to a smaller subset (20 features and 15 features), its performance will drop from 97% to 96% and 95% respectively. On the other hand, the T-test filter method cannot provide an effective feature subset given a dataset with highly correlated features.

The feature subsets acquired from embedded feature selection approaches, both random forest, and lasso penalized logistic regression are relatively larger, and smaller subsets can result in the cost of lower performance. Based on our experiments, embedded feature selection approaches cannot only compete with wrapper methods and information gain filter method.
