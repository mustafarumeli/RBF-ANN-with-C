#include <stdio.h>
#include <gsl\gsl_linalg.h>
#include <string.h>
#include <stdlib.h>
#define MAX 863
#define SUPERMAX 744769
#define CLUSTERCOUNT 45
#define SUPERCLUSTERCOUNT (CLUSTERCOUNT * 2 + 1)
#include <math.h>

typedef struct Node //We constructed a struct for each point  and center 
{
	double X;
	double Y;
} Node;
typedef struct Centroid //For differ centroid 
{
	double X;
	double Y;
} Centroid;

double** initiateMatrix(int x, int y); //For simplifing initiate matrices we created this function 
Centroid* Kmeans(Node* X, int xDataSize, Centroid* initial_centroids, int max_iter); //For calculating KMeans 
int FindMin(double arry[MAX], int size);//For finding minimum of an array
double* ComputeRBFBetas(Node* X, int xCount, Centroid* centroids, int* membership); // For calculating Betas
double Mean(double distances[MAX], int size); // for calculating mean of an array
int GetDistinct(int arry[MAX], int size);// for finding number of  distinct elements
double* GetRBFActivations(Centroid* centerFull, double* betaFull, Node currentNode); // For finding neuron activation threshold
int HasElement(int arry[MAX], int size, int val); // for finding wheter given array has given value 
double** GetTranspose(double** theta, int xCount, int yCount); // for generating transpose of given matrix
int* findClosestCentroids(Node* data, int m, Centroid* centroids, int k); // For each node calculates nodes membership to centroids
double** GetTransposeArray(double* theta, int xCount); // for generating transpose of given array
double** MatrixMultiply(double** first, double** second, int m, int q, int np);// for multiplying given two matricies
int CheckIfCentroidsAreEqual(Centroid* first, Centroid* second, int size);// Cheks if the given two centroid arrays are same
double** inverse(double** a, int k); // For inversing  the given matrix
Centroid* ComputeCentroids(Node* X, int xDataSize, Centroid* prevCentroids, int* memberships, int k); // positioning the centroids 
void MakeTwoCentroidsEqual(Centroid* first, Centroid* second, int size);// equalizes two centroid arrays
Centroid FindMeanForNodes(Node* nodes, int nodeSize); // for finding mean of given node array
double* EvaluateRBFN(Centroid* centers, double* betas, double** theta, Node input); // Computes the outputs of an RBF Network for the provided input.
double** RbfTrainer(Node xTrain[MAX], int yTrain[MAX], int centersPerCat, Centroid*  centerFull, double* betaFull); // for training the neural network 
double GetEuclideanDistanceSquared(Node node, Centroid center); // for calculating EuclideanDistance between given node and centroid
double GetGaussianResult(double beta, Centroid center, Node node); // for calculating GaussianResult
Centroid* PrepareInitialCentroids(Node* nodes); // for PreparingInitialCentroids for kmeans and rbf


double GetEuclideanDistanceSquared(Node node, Centroid center) {
	double xDistance = node.X - center.X;
	xDistance *= xDistance;
	double yDistance = node.Y - center.Y;
	yDistance *= yDistance;
	return xDistance + yDistance;
}
double GetGaussianResult(double beta, Centroid center, Node node) {
	return exp(-1 * beta*GetEuclideanDistanceSquared(node, center));
}
Centroid* Kmeans(Node* X, int xDataSize, Centroid* initial_centroids, int max_iter) {
	int k = CLUSTERCOUNT;
	Centroid* centroids = malloc(k * sizeof(Centroid));
	Centroid* prevCentroids = malloc(k * sizeof(Centroid));
	MakeTwoCentroidsEqual(initial_centroids, centroids, k);
	MakeTwoCentroidsEqual(centroids, prevCentroids, k);

	for (size_t i = 0; i < max_iter; i++)
	{
		int* membership = malloc(MAX * sizeof(int));
		membership = findClosestCentroids(X, xDataSize, centroids, CLUSTERCOUNT);
		Centroid sa[CLUSTERCOUNT];
		for (size_t i = 0; i < CLUSTERCOUNT; i++)
		{
			sa[i] = centroids[i];
		}
		centroids = ComputeCentroids(X, xDataSize, centroids, membership, CLUSTERCOUNT);
		Centroid sa2[CLUSTERCOUNT];
		for (size_t i = 0; i < CLUSTERCOUNT; i++)
		{
			sa2[i] = centroids[i];
		}
		if (CheckIfCentroidsAreEqual(centroids, prevCentroids, CLUSTERCOUNT) == 1)
		{
			break;
		}

		MakeTwoCentroidsEqual(centroids, prevCentroids, CLUSTERCOUNT);
	}
	return centroids;

}
void MakeTwoCentroidsEqual(Centroid* first, Centroid* second, int size) {
	for (size_t i = 0; i < size; i++)
	{
		second[i].X = first[i].X;
		second[i].Y = first[i].Y;
	}
}
int CheckIfCentroidsAreEqual(Centroid* first, Centroid* second, int size) {
	for (size_t i = 0; i < size; i++)
	{
		if (first[i].X != second[i].X || first[i].Y != second[i].Y)
		{
			return 0;
		}
	}
	return 1;
}
Centroid* ComputeCentroids(Node* X, int xDataSize, Centroid* prevCentroids, int* memberships, int k) {
	Centroid* centroid = malloc(CLUSTERCOUNT * sizeof(Centroid));
	for (size_t i = 0; i < CLUSTERCOUNT; i++)
	{
		centroid[i].X = 0;
		centroid[i].Y = 0;
	}

	for (size_t i = 0; i < k; i++)
	{
		if (HasElement(memberships, xDataSize, i) == 0)
		{
			centroid[i].X = prevCentroids[i].X;
			centroid[i].Y = prevCentroids[i].Y;
		}
		else
		{
			Node* points = malloc(xDataSize * sizeof(Node));
			int pointCounter = 0;
			for (size_t t = 0; t < xDataSize; t++)
			{
				if (memberships[t] == i)
				{
					points[pointCounter].X = X[t].X;
					points[pointCounter].Y = X[t].Y;
					pointCounter++;
				}
			}
			Centroid cNow = FindMeanForNodes(points, pointCounter);
			centroid[i].X = cNow.X;
			centroid[i].Y = cNow.Y;
		}
	}
	return centroid;
}
int* findClosestCentroids(Node* data, int m, Centroid* centroids, int k) {
	int* membership = malloc(m * sizeof(int));
	double** distances = initiateMatrix(m, k);
	for (size_t i = 0; i < m; i++)
	{
		membership[i] = 0;
	}

	for (int j = 0; j < k; j++)
	{
		Centroid currentCenter = centroids[j];
		for (int i = 0; i < m; i++)
		{
			Node currentNode = data[i];
			distances[i][j] = GetEuclideanDistanceSquared(currentNode, currentCenter);
		}
	}

	for (size_t z = 0; z < m; z++)
	{
		membership[z] = FindMin(distances[z], k);
	}
	return membership;
}
Centroid FindMeanForNodes(Node* nodes, int nodeSize) {
	double xSum = 0, ySum = 0;
	for (size_t i = 0; i < nodeSize; i++)
	{
		xSum += nodes[i].X;
		ySum += nodes[i].Y;
	}
	Centroid c;
	c.X = (xSum / (double)nodeSize);
	c.Y = (ySum / (double)nodeSize);
	return c;
}
int FindMin(double arry[MAX], int size) {
	double min = DBL_MAX;
	int minIndex = 0;
	for (int i = 0; i < size; i++)
	{
		if (min > arry[i])
		{
			min = arry[i];
			minIndex = i;
		}
	}
	return minIndex;
}
Centroid* PrepareInitialCentroids(Node* nodes)
{
	Centroid* init_centroids = malloc(CLUSTERCOUNT * sizeof(Centroid));
	for (size_t i = 0; i < CLUSTERCOUNT; i++)
	{
		init_centroids[i].X = nodes[i].X;
		init_centroids[i].Y = nodes[i].Y;
	}
	return init_centroids;
}
double** RbfTrainer(Node xTrain[MAX], int yTrain[MAX], int centersPerCat, Centroid*  centerFull, double* betaFull) {
	int* ycFull = malloc(MAX * sizeof(int));

	int superCounter = 0;
	int numberOfCats = GetDistinct(yTrain, MAX);
	Node* XC1 = malloc(MAX * sizeof(Node));
	int xc1Counter = 0, xc2Counter = 0;
	Node* XC2 = malloc(MAX * sizeof(Node));
	double** yc1 = initiateMatrix(MAX, 1);
	double** yc2 = initiateMatrix(MAX, 1);

	for (size_t j = 0; j < MAX; j++)
	{
		if (yTrain[j] == 1)
		{
			XC1[xc1Counter] = xTrain[j];
			yc1[j][0] = 1;
			yc2[j][0] = 0;
			xc1Counter++;
		}
		else
		{
			XC2[xc2Counter] = xTrain[j];
			yc1[j][0] = 0;
			yc2[j][0]= 1;
			xc2Counter++;
		}
	}

	Centroid* centro = PrepareInitialCentroids(XC1);// = (Centroid*)malloc(10 * sizeof(Centroid));
	Centroid* newCentro = Kmeans(XC1, xc1Counter, centro, 100);
	int* membership = malloc(MAX * sizeof(int));
	membership = findClosestCentroids(XC1, xc1Counter, newCentro, CLUSTERCOUNT);
	double* betas = malloc(sizeof(double) * CLUSTERCOUNT);
	betas = ComputeRBFBetas(XC1, xc1Counter, newCentro, membership);

	Centroid* centro2 = PrepareInitialCentroids(XC2);// = (Centroid*)malloc(10 * sizeof(Centroid));
	Centroid* newCentro2 = Kmeans(XC2, xc2Counter, centro2, 100);
	int* membership2 = malloc(MAX * sizeof(int));
	membership2 = findClosestCentroids(XC2, xc2Counter, newCentro2, CLUSTERCOUNT);
	double* betas2 = malloc(sizeof(double) * CLUSTERCOUNT);
	betas2 = ComputeRBFBetas(XC2, xc2Counter, newCentro2, membership2);


	for (size_t k = 0; k < CLUSTERCOUNT; k++)
	{
		betaFull[k] = betas[k];
		betaFull[k + CLUSTERCOUNT] = betas2[k];
		centerFull[k] = newCentro[k];
		centerFull[k + CLUSTERCOUNT] = newCentro2[k];
	}
	for (size_t i = 0; i < xc1Counter; ycFull[i] = yc1[i++]);
	for (size_t i = xc2Counter, j = 0; i < MAX; ycFull[i++] = yc2[j++]);

	//double** x_active = initiateMatrix(MAX, 21);
	double** x_active = initiateMatrix(MAX, SUPERCLUSTERCOUNT);
	for (size_t counter = 0; counter < MAX; counter++)
	{
		double* z = malloc(sizeof(double) * CLUSTERCOUNT * 2);
		z = GetRBFActivations(centerFull, betaFull, xTrain[counter]);
		double* zTranspose = GetTransposeArray(z, CLUSTERCOUNT * 2);
		for (size_t i = 0; i < CLUSTERCOUNT * 2; i++)
		{
			x_active[counter][i] = 1;
		}
		for (size_t i = 0; i < CLUSTERCOUNT * 2; i++)
		{
			x_active[counter][i + 1] = z[i];
		}

	}
	double** transposed = GetTranspose(x_active, MAX, SUPERCLUSTERCOUNT);
	double** multiplied = MatrixMultiply(transposed, x_active, SUPERCLUSTERCOUNT, SUPERCLUSTERCOUNT, MAX);
	double** pinv = inverse(multiplied, SUPERCLUSTERCOUNT);
	double** pinbS = MatrixMultiply(pinv, transposed,SUPERCLUSTERCOUNT,MAX,SUPERCLUSTERCOUNT);
	
	double** theta1 = MatrixMultiply(pinbS, yc1, SUPERCLUSTERCOUNT, 1, MAX);
	double** theta2 = MatrixMultiply(pinbS, yc2, SUPERCLUSTERCOUNT, 1, MAX);
	double** theta = initiateMatrix(SUPERCLUSTERCOUNT, 2);
	for (size_t i = 0; i < SUPERCLUSTERCOUNT; i++)
	{
		theta[i][0] = theta1[i][0];
		theta[i][1] = theta2[i][0];
	}

	return theta;
}
double** initiateMatrix(int x, int y) {
	double** matrix = malloc(x * sizeof(double*));
	for (size_t i = 0; i < x; i++)
	{
		matrix[i] = malloc(y * sizeof(double));
	}
	return matrix;
}
double** inverse(double** a, int k) {
	int s;
	double** inversed = initiateMatrix(k, k);
	double* superA = malloc(k*k * sizeof(double));
	double* superInversed = malloc(k*k * sizeof(double));
	int superCounter = 0;
	for (size_t i = 0; i < k; i++)
	{
		for (size_t j = 0; j < k; j++)
		{
			superA[superCounter] = a[i][j];
			superInversed[superCounter] = 0;
			superCounter++;

		}
	}
	gsl_matrix_view m
		= gsl_matrix_view_array(superA, k, k);
	gsl_matrix_view inv
		= gsl_matrix_view_array(superInversed, k, k);
	gsl_permutation * p = gsl_permutation_alloc(k);



	gsl_linalg_LU_decomp(&m.matrix, p, &s);
	gsl_linalg_LU_invert(&m.matrix, p, &inv.matrix);
	for (int i = 0; i < k; ++i) {
		for (int j = 0; j < k; ++j) {
			double val = gsl_matrix_get(&inv.matrix, i, j);
			inversed[i][j] = val;
		}
	}


	gsl_permutation_free(p);
	return inversed;
}
double** MatrixMultiply(double** first, double** second, int m, int q,int np) {

	double** multiply = initiateMatrix(m, q);
	double sum = 0;
	int i, j, k;
	for (i = 0; i < m; i++)
	{
		for (j = 0; j < q; j++)
		{
			multiply[i][j] = 0;
			for (k = 0; k < np; k++)
			{
				multiply[i][j] += first[i][k] * second[k][j];
			}
		}
	}
	return multiply;
}
double* ComputeRBFBetas(Node* X, int xCount, Centroid* centroids, int* membership) {
	double* sigmas = malloc(CLUSTERCOUNT * sizeof(double));
	for (size_t i = 0; i < CLUSTERCOUNT; i++)
	{
		sigmas[i] = 0;
	}

	for (size_t i = 0; i < CLUSTERCOUNT; i++)
	{
		Centroid currentCentroid = centroids[i];
		double* differs = malloc(xCount * sizeof(double));
		int memberCount = 0;
		for (size_t t = 0; t < xCount; t++)
		{
			if (membership[t] == i)
			{

				differs[memberCount] = sqrt(GetEuclideanDistanceSquared(X[t], currentCentroid));
				memberCount++;
			}
		}
		sigmas[i] = Mean(differs, memberCount);
	}
	double* betas = malloc(CLUSTERCOUNT * sizeof(double));
	for (size_t i = 0; i < CLUSTERCOUNT; i++)
	{
		betas[i] = 1 / (2 * sigmas[i] * sigmas[i]);
	}
	return betas;
}
double Mean(double distances[MAX], int size) {
	double sum = 0;
	for (size_t i = 0; i < size; i++)
	{
		sum += distances[i];
	}
	return (double)sum / size;
}
int GetDistinct(int arry[MAX], int size) {
	int* rtnArry = (int*)malloc(size * sizeof(int));
	int c = 0;
	for (int i = 0; i < size; i++)
	{
		if (HasElement(rtnArry, size, arry[i]) == 0)
		{
			rtnArry[c] = arry[i];
			c++;
		}
	}
	return c;
}
int HasElement(int arry[MAX], int size, int val) {
	for (size_t i = 0; i < size; i++)
	{
		if (arry[i] == val)
		{
			return 1;
		}
	}
	return 0;
}
int main()
{

	char buffer[1024];
	char *record, *line;
	int i = 0;
	Node nodes[MAX];
	int  classify[MAX];
	FILE *fstream = fopen("C:\\Users\\zuzu\\Desktop\\RBFN Example\\dataset.csv", "r");
	if (fstream == NULL)
	{
		printf("\n file opening failed ");
		return -1;
	}
	while ((line = fgets(buffer, sizeof(buffer), fstream)) != NULL && i <= MAX)
	{
		record = strtok(line, ",");
		int c = 0;
		//printf("%d\n", i);    //here you can put the record into the array as per your requirement.
		while (record != NULL)
		{

			if (c < 2)
			{
				if (c == 0)
				{
					nodes[i].X = atof(record);
				}
				else
				{
					nodes[i].Y = atof(record);
				}
			}
			else
			{
				classify[i] = atoi(record);
			}
			record = strtok(NULL, ",");
			c++;
		}
		i++;
		if (i == MAX)
		{
			break;
		}
	}
	double* betas = malloc(CLUSTERCOUNT * 2 * sizeof(double));
	Centroid* centers = malloc(CLUSTERCOUNT * 2 * sizeof(Centroid));
	double** theta = RbfTrainer(nodes, classify, CLUSTERCOUNT, centers, betas);
	int correctCount = 0;
	Node ErrorNodes[MAX];// = malloc(sizeof(Node) * MAX);
	int errorCounter = 0;
	for (size_t i = 0; i < MAX; i++)
	{
		double* scores = EvaluateRBFN(centers, betas, theta, nodes[i]);
		if (scores[0] > scores[1])
		{
			if (classify[i] == 1)
			{
				correctCount++;
			}
			else
			{
				ErrorNodes[errorCounter++] = nodes[i];
			}
		}
		else
		{
			if (classify[i] == 2)
			{
				correctCount++;
			}
			else
			{
				ErrorNodes[errorCounter++] = nodes[i];
			}
		}

	}
	double acc = ((double)correctCount / MAX);
	printf("ACC: %f", acc * 100.0);
	getch();
	return 0;
}
double* EvaluateRBFN(Centroid* centers, double* betas, double** theta, Node input) {
	double* phisTemp = GetRBFActivations(centers, betas, input);
	double** phis = initiateMatrix(SUPERCLUSTERCOUNT, 1);
	phis[0][0]= 1;
	for (size_t i = 1, phisTempCounter = 0; i < SUPERCLUSTERCOUNT; i++, phisTempCounter++)
	{
		phis[i][0] = phisTemp[phisTempCounter];
	}
	double** thetaTranspose = GetTranspose(theta, SUPERCLUSTERCOUNT, 2);
	double** tempZ = MatrixMultiply(thetaTranspose, phis, 2, 1, SUPERCLUSTERCOUNT);
	double z[2];
	z[0] = tempZ[0][0];
	z[1] = tempZ[1][0];
	return z;
}
double* GetRBFActivations(Centroid* centerFull, double* betaFull, Node currentNode) {
	double* z = malloc(CLUSTERCOUNT * 2 * sizeof(double));
	for (size_t i = 0; i < CLUSTERCOUNT * 2; i++)
	{
		z[i] = GetGaussianResult(betaFull[i], centerFull[i], currentNode);
	}
	return z;
}
double** GetTransposeArray(double* theta, int xCount) {
	double** transposed = initiateMatrix(1,xCount);
	for (int i = 0; i < xCount; ++i)
	{
		transposed[0][i] = theta[i];
	}
		
	return transposed;
}
double** GetTranspose(double** theta,int xCount,int yCount) {
	double** transposed = initiateMatrix(yCount, xCount);
	for (int i = 0; i < xCount; ++i)
		for (int j = 0; j < yCount; ++j)
		{
			transposed[j][i] = theta[i][j];
		}
	return transposed;
}
