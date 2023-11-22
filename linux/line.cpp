/*
This is the tool ....

Contact Author: Jian Tang, Microsoft Research, jiatang@microsoft.com, tangjianpku@gmail.com
Publication: Jian Tang, Meng Qu, Mingzhe Wang, Ming Zhang, Jun Yan, Qiaozhu Mei. "LINE: Large-scale Information Network Embedding". In WWW 2015.
*/

// Format of the training file:
//
// The training file contains serveral lines, each line represents a DIRECTED edge in the network.
// More specifically, each line has the following format "<u> <v> <w>", meaning an edge from <u> to <v> with weight as <w>.
// <u> <v> and <w> are seperated by ' ' or '\t' (blank or tab)
// For UNDIRECTED edge, the user should use two DIRECTED edges to represent it.


#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <pthread.h>
#include <gsl/gsl_rng.h>


#define MAX_STRING 100
#define SIGMOID_BOUND 6
#define NEG_SAMPLING_POWER 0.75

const int hash_table_size = 30000000;  // 30 million
const int neg_table_size = 1e8; // 100 million
const int sigmoid_table_size = 1000;

typedef float real;                    // Precision of float numbers

struct ClassVertex {  //representing vertex of a network，为什么degree是一个double值？
	double degree;
	char *name;
};

char network_file[MAX_STRING], embedding_file[MAX_STRING];  //输入文件和输出文件的文件名最多100字符
struct ClassVertex *vertex; //vertex consists of {name, degree}
int is_binary = 0, num_threads = 1, order = 2, dim = 100, num_negative = 5;  // command line arguments and default values

//vertex_hash_table是一个哈希表，其地址是顶点name的hash值，其值是顶点在 *vertex 顶点列表的下标
int *vertex_hash_table, *neg_table;

//num_vertices用于记录 *vertex数组 已经存入了多少个顶点，即下一个顶点应存放的下标
//max_num_vertices用于设置 *vertex数组 的大小，当数组快满时，max_num_vertices自增1000，然后重新分配空间realloc
int max_num_vertices = 1000, num_vertices = 0;

// total_samples: total number of samples
// current_samples: current processing number of samples
// num_edges: 啥意思？
long long total_samples = 1, current_sample_count = 0, num_edges = 0;

real init_rho = 0.025, rho;  //初始学习率和学习率
real *emb_vertex, *emb_context, *sigmoid_table;  //某顶点的嵌入向量，嵌入上下文向量（2nd order），sigmoid表


//这里是处理输入文件所使用的变量
// edge_source_id: 边源节点向量
// edge_target_id: 边目标节点向量
// edge_weight: 边权重向量
int *edge_source_id, *edge_target_id;
double *edge_weight;

// Parameters for edge sampling
// 边缘概率分布？ 边缘采样相关参数
long long *alias;  //顶点的别名向量
double *prob;  //顶点的边缘概率向量


/*
typedef struct
  {
    const char *name;
    unsigned long int max;
    unsigned long int min;
    size_t size;

	// 下面的set和get，get_double都是函数指针
	// set: 输入参数state，seed，返回值为void
	// get: 输入参数state，返回int
	// get_double: 输入参数state，返回double
    void (*set) (void *state, unsigned long int seed);  
    unsigned long int (*get) (void *state);
    double (*get_double) (void *state);
  } gsl_rng_type;
*/
const gsl_rng_type * gsl_T;

/*
typedef struct
  {
    const gsl_rng_type * type;
    void *state;  //void指针可以存放任意类型的地址，且无需进行类型转换；如果要把void指针类型赋值给别人时，需要进行强制类型转换
  } gsl_rng; 
*/
gsl_rng * gsl_r;

/* Build a hash table, mapping each vertex name to a unique vertex id(unsigned int value) */
// Hash: string -> unsigned int(上界为3千万)
unsigned int Hash(char *key)
{
	unsigned int seed = 131;
	unsigned int hash = 0;
	while (*key)
	{
		hash = hash * seed + (*key++);
	}
	return hash % hash_table_size;  //hash值最大为 3千万, vertex 数量应远小于 3千万
}

// vertex_hash_table是一个长度为3千万的int数组，所有位置初始值置为-1
// vertex_hash_table表示一个哈希表，其地址是顶点name的hash值，其值是顶点在 *vertex 顶点列表的下标
void InitHashTable()
{
	vertex_hash_table = (int *)malloc(hash_table_size * sizeof(int));
	for (int k = 0; k != hash_table_size; k++) vertex_hash_table[k] = -1;
}

//将指定的value插入到key对应的位置（地址addr）中
//这里的value应该是顶点在 *vertex数组 的下标
void InsertHashTable(char *key, int value)
{
	int addr = Hash(key);  //1. 先计算出key在hash_table中的位置（地址addr）
	while (vertex_hash_table[addr] != -1){  //2. 如果对应位置已经有值了，则顺延到下一个位置（这是一个不可删除元素的hash_table!!!）
		addr = (addr + 1) % hash_table_size;
	}
	vertex_hash_table[addr] = value;  //3. 令对应位置的值为给定的value
}

//根据指定的key搜索hash_table中对应的值
int SearchHashTable(char *key)
{
	int addr = Hash(key);  //1. 先计算出key在hash_table中的位置（地址addr）
	while (1){
		if (vertex_hash_table[addr] == -1) return -1;  //2.1 如果对应位置没有值，返回-1，说明key没有录入hash_table
		//2.2 如果对应位置有值，查看此值对应的 ClassVertex变量 的名字是否等于key，若是，返回此value(即顶点在 *vertex数组 中的下标)
		if (!strcmp(key, vertex[vertex_hash_table[addr]].name)) return vertex_hash_table[addr];
		//3. 否则查找下一个位置
		addr = (addr + 1) % hash_table_size;

		//hash_table不可以装满，即图的顶点数必须小于3千万，否则会产生死循环
	}
	return -1;
}

/* Add a vertex to the vertex set */
int AddVertex(char *name)
{
	//1. 如果顶点name大于100字符，截取其前100字符
	int length = strlen(name) + 1;
	if (length > MAX_STRING) length = MAX_STRING;

	//2. 在 *vertex数组中， 加入新的顶点，令其name为传入的name，degree为0
	vertex[num_vertices].name = (char *)calloc(length, sizeof(char));
	strncpy(vertex[num_vertices].name, name, length-1);
	vertex[num_vertices].degree = 0;
	num_vertices++;

	//3. 如果 *vertex数组大小快不够了，重新分配 *vertex空间，增加1000个位置
	if (num_vertices + 2 >= max_num_vertices)
	{
		max_num_vertices += 1000;
		vertex = (struct ClassVertex *)realloc(vertex, max_num_vertices * sizeof(struct ClassVertex));
	}

	//4. 将对应的name，*vertex数组下标 存放到hash_table中方面随机存取
	InsertHashTable(name, num_vertices - 1);
	//5. 返回值为加入到顶点在 *vertex数组 中的下标
	return num_vertices - 1; 
}

/* Read network from the training file */
void ReadData()
{
	FILE *fin;
	char name_v1[MAX_STRING], name_v2[MAX_STRING], str[2 * MAX_STRING + 10000];
	int vid;
	double weight;

	//1. 第一次读二进制文件network_file, 数边的数量，进行相应的初始化
	fin = fopen(network_file, "rb");
	if (fin == NULL)
	{
		printf("ERROR: network file not found!\n");
		exit(1);
	}
	num_edges = 0; //初始化边的数量为0
	//一次读一行，边数量num_edges++
	while (fgets(str, sizeof(str), fin)){
		num_edges++;
	}
	fclose(fin);
	printf("Number of edges: %lld          \n", num_edges);
	edge_source_id = (int *)malloc(num_edges*sizeof(int));
	edge_target_id = (int *)malloc(num_edges*sizeof(int));
	edge_weight = (double *)malloc(num_edges*sizeof(double));
	if (edge_source_id == NULL || edge_target_id == NULL || edge_weight == NULL)
	{
		printf("Error: memory allocation failed!\n");
		exit(1);
	}

	//2. 第二次读二进制文件network_file
	fin = fopen(network_file, "rb");
	num_vertices = 0;
	for (int k = 0; k != num_edges; k++)
	{
		fscanf(fin, "%s %s %lf", name_v1, name_v2, &weight);

		if (k % 10000 == 0)
		{
			printf("Reading edges: %.3lf%%%c", k / (double)(num_edges + 1) * 100, 13);
			fflush(stdout);
		}

		vid = SearchHashTable(name_v1);
		if (vid == -1) vid = AddVertex(name_v1);
		vertex[vid].degree += weight;
		edge_source_id[k] = vid;

		vid = SearchHashTable(name_v2);
		if (vid == -1) vid = AddVertex(name_v2);
		vertex[vid].degree += weight;
		edge_target_id[k] = vid;

		edge_weight[k] = weight;
	}
	fclose(fin);
	printf("Number of vertices: %d          \n", num_vertices);
}

/* The alias sampling algorithm, which is used to sample an edge in O(1) time. */
void InitAliasTable()
{
	alias = (long long *)malloc(num_edges*sizeof(long long));
	prob = (double *)malloc(num_edges*sizeof(double));
	if (alias == NULL || prob == NULL)
	{
		printf("Error: memory allocation failed!\n");
		exit(1);
	}

	double *norm_prob = (double*)malloc(num_edges*sizeof(double));
	long long *large_block = (long long*)malloc(num_edges*sizeof(long long));
	long long *small_block = (long long*)malloc(num_edges*sizeof(long long));
	if (norm_prob == NULL || large_block == NULL || small_block == NULL)
	{
		printf("Error: memory allocation failed!\n");
		exit(1);
	}

	double sum = 0;
	long long cur_small_block, cur_large_block;
	long long num_small_block = 0, num_large_block = 0;

	for (long long k = 0; k != num_edges; k++) sum += edge_weight[k];
	for (long long k = 0; k != num_edges; k++) norm_prob[k] = edge_weight[k] * num_edges / sum;

	for (long long k = num_edges - 1; k >= 0; k--)
	{
		if (norm_prob[k]<1)
			small_block[num_small_block++] = k;
		else
			large_block[num_large_block++] = k;
	}

	while (num_small_block && num_large_block)
	{
		cur_small_block = small_block[--num_small_block];
		cur_large_block = large_block[--num_large_block];
		prob[cur_small_block] = norm_prob[cur_small_block];
		alias[cur_small_block] = cur_large_block;
		norm_prob[cur_large_block] = norm_prob[cur_large_block] + norm_prob[cur_small_block] - 1;
		if (norm_prob[cur_large_block] < 1)
			small_block[num_small_block++] = cur_large_block;
		else
			large_block[num_large_block++] = cur_large_block;
	}

	while (num_large_block) prob[large_block[--num_large_block]] = 1;
	while (num_small_block) prob[small_block[--num_small_block]] = 1;

	free(norm_prob);
	free(small_block);
	free(large_block);
}

long long SampleAnEdge(double rand_value1, double rand_value2)
{
	long long k = (long long)num_edges * rand_value1;
	return rand_value2 < prob[k] ? k : alias[k];
}

/* Initialize the vertex embedding and the context embedding */
void InitVector()
{
	long long a, b;

	a = posix_memalign((void **)&emb_vertex, 128, (long long)num_vertices * dim * sizeof(real));
	if (emb_vertex == NULL) { printf("Error: memory allocation failed\n"); exit(1); }
	for (b = 0; b < dim; b++) for (a = 0; a < num_vertices; a++)
		emb_vertex[a * dim + b] = (rand() / (real)RAND_MAX - 0.5) / dim;

	a = posix_memalign((void **)&emb_context, 128, (long long)num_vertices * dim * sizeof(real));
	if (emb_context == NULL) { printf("Error: memory allocation failed\n"); exit(1); }
	for (b = 0; b < dim; b++) for (a = 0; a < num_vertices; a++)
		emb_context[a * dim + b] = 0;
}

/* Sample negative vertex samples according to vertex degrees */
void InitNegTable()
{
	double sum = 0, cur_sum = 0, por = 0;
	int vid = 0;
	neg_table = (int *)malloc(neg_table_size * sizeof(int));
	for (int k = 0; k != num_vertices; k++) sum += pow(vertex[k].degree, NEG_SAMPLING_POWER);
	for (int k = 0; k != neg_table_size; k++)
	{
		if ((double)(k + 1) / neg_table_size > por)
		{
			cur_sum += pow(vertex[vid].degree, NEG_SAMPLING_POWER);
			por = cur_sum / sum;
			vid++;
		}
		neg_table[k] = vid - 1;
	}
}

/* Fastly compute sigmoid function */
void InitSigmoidTable()
{
	real x;
	sigmoid_table = (real *)malloc((sigmoid_table_size + 1) * sizeof(real));
	for (int k = 0; k != sigmoid_table_size; k++)
	{
		x = 2.0 * SIGMOID_BOUND * k / sigmoid_table_size - SIGMOID_BOUND;
		sigmoid_table[k] = 1 / (1 + exp(-x));
	}
}

real FastSigmoid(real x)
{
	if (x > SIGMOID_BOUND) return 1;
	else if (x < -SIGMOID_BOUND) return 0;
	int k = (x + SIGMOID_BOUND) * sigmoid_table_size / SIGMOID_BOUND / 2;
	return sigmoid_table[k];
}

/* Fastly generate a random integer */
int Rand(unsigned long long &seed)
{
	seed = seed * 25214903917 + 11;
	return (seed >> 16) % neg_table_size;
}

/* Update embeddings */
void Update(real *vec_u, real *vec_v, real *vec_error, int label)
{
	real x = 0, g;
	for (int c = 0; c != dim; c++) x += vec_u[c] * vec_v[c];
	g = (label - FastSigmoid(x)) * rho;
	for (int c = 0; c != dim; c++) vec_error[c] += g * vec_v[c];
	for (int c = 0; c != dim; c++) vec_v[c] += g * vec_u[c];
}

void *TrainLINEThread(void *id)
{
	long long u, v, lu, lv, target, label;
	long long count = 0, last_count = 0, curedge;
	unsigned long long seed = (long long)id;
	real *vec_error = (real *)calloc(dim, sizeof(real));

	while (1)
	{
		//judge for exit
		if (count > total_samples / num_threads + 2) break;

		if (count - last_count > 10000)
		{
			current_sample_count += count - last_count;
			last_count = count;
			printf("%cRho: %f  Progress: %.3lf%%", 13, rho, (real)current_sample_count / (real)(total_samples + 1) * 100);
			fflush(stdout);
			rho = init_rho * (1 - current_sample_count / (real)(total_samples + 1));
			if (rho < init_rho * 0.0001) rho = init_rho * 0.0001;
		}

		curedge = SampleAnEdge(gsl_rng_uniform(gsl_r), gsl_rng_uniform(gsl_r));
		u = edge_source_id[curedge];
		v = edge_target_id[curedge];

		lu = u * dim;
		for (int c = 0; c != dim; c++) vec_error[c] = 0;

		// NEGATIVE SAMPLING
		for (int d = 0; d != num_negative + 1; d++)
		{
			if (d == 0)
			{
				target = v;
				label = 1;
			}
			else
			{
				target = neg_table[Rand(seed)];
				label = 0;
			}
			lv = target * dim;
			if (order == 1) Update(&emb_vertex[lu], &emb_vertex[lv], vec_error, label);
			if (order == 2) Update(&emb_vertex[lu], &emb_context[lv], vec_error, label);
		}
		for (int c = 0; c != dim; c++) emb_vertex[c + lu] += vec_error[c];

		count++;
	}
	free(vec_error);
	pthread_exit(NULL);
}

void Output()
{
	FILE *fo = fopen(embedding_file, "wb");
	fprintf(fo, "%d %d\n", num_vertices, dim);
	for (int a = 0; a < num_vertices; a++)
	{
		fprintf(fo, "%s ", vertex[a].name);
		if (is_binary) for (int b = 0; b < dim; b++) fwrite(&emb_vertex[a * dim + b], sizeof(real), 1, fo);
		else for (int b = 0; b < dim; b++) fprintf(fo, "%lf ", emb_vertex[a * dim + b]);
		fprintf(fo, "\n");
	}
	fclose(fo);
}

void TrainLINE() {
	long a;
	pthread_t *pt = (pthread_t *)malloc(num_threads * sizeof(pthread_t));

	if (order != 1 && order != 2)
	{
		printf("Error: order should be either 1 or 2!\n");
		exit(1);
	}
	printf("--------------------------------\n");
	printf("Order: %d\n", order);
	printf("Samples: %lldM\n", total_samples / 1000000);
	printf("Negative: %d\n", num_negative);
	printf("Dimension: %d\n", dim);
	printf("Initial rho: %lf\n", init_rho);
	printf("--------------------------------\n");

	InitHashTable();
	ReadData();
	InitAliasTable();
	InitVector();
	InitNegTable();
	InitSigmoidTable();

	gsl_rng_env_setup();
	gsl_T = gsl_rng_rand48;
	gsl_r = gsl_rng_alloc(gsl_T);
	gsl_rng_set(gsl_r, 314159265);

	clock_t start = clock();
	printf("--------------------------------\n");
	for (a = 0; a < num_threads; a++) pthread_create(&pt[a], NULL, TrainLINEThread, (void *)a);
	for (a = 0; a < num_threads; a++) pthread_join(pt[a], NULL);
	printf("\n");
	clock_t finish = clock();
	printf("Total time: %lf\n", (double)(finish - start) / CLOCKS_PER_SEC);

	Output();
}

int ArgPos(char *str, int argc, char **argv) {
	int a;
	for (a = 1; a < argc; a++) if (!strcmp(str, argv[a])) {
		if (a == argc - 1) {
			printf("Argument missing for %s\n", str);
			exit(1);
		}
		return a;
	}
	return -1;
}

int main(int argc, char **argv) {
	int i;
	if (argc == 1) {
		printf("LINE: Large Information Network Embedding\n\n");
		printf("Options:\n");
		printf("Parameters for training:\n");
		printf("\t-train <file>\n");
		printf("\t\tUse network data from <file> to train the model\n");
		printf("\t-output <file>\n");
		printf("\t\tUse <file> to save the learnt embeddings\n");
		printf("\t-binary <int>\n");
		printf("\t\tSave the learnt embeddings in binary moded; default is 0 (off)\n");
		printf("\t-size <int>\n");
		printf("\t\tSet dimension of vertex embeddings; default is 100\n");
		printf("\t-order <int>\n");
		printf("\t\tThe type of the model; 1 for first order, 2 for second order; default is 2\n");
		printf("\t-negative <int>\n");
		printf("\t\tNumber of negative examples; default is 5\n");
		printf("\t-samples <int>\n");
		printf("\t\tSet the number of training samples as <int>Million; default is 1\n");
		printf("\t-threads <int>\n");
		printf("\t\tUse <int> threads (default 1)\n");
		printf("\t-rho <float>\n");
		printf("\t\tSet the starting learning rate; default is 0.025\n");
		printf("\nExamples:\n");
		printf("./line -train net.txt -output vec.txt -binary 1 -size 200 -order 2 -negative 5 -samples 100 -rho 0.025 -threads 20\n\n");
		return 0;
	}
	if ((i = ArgPos((char *)"-train", argc, argv)) > 0) strcpy(network_file, argv[i + 1]);
	if ((i = ArgPos((char *)"-output", argc, argv)) > 0) strcpy(embedding_file, argv[i + 1]);
	if ((i = ArgPos((char *)"-binary", argc, argv)) > 0) is_binary = atoi(argv[i + 1]);
	if ((i = ArgPos((char *)"-size", argc, argv)) > 0) dim = atoi(argv[i + 1]);
	if ((i = ArgPos((char *)"-order", argc, argv)) > 0) order = atoi(argv[i + 1]);
	if ((i = ArgPos((char *)"-negative", argc, argv)) > 0) num_negative = atoi(argv[i + 1]);
	if ((i = ArgPos((char *)"-samples", argc, argv)) > 0) total_samples = atoi(argv[i + 1]);
	if ((i = ArgPos((char *)"-rho", argc, argv)) > 0) init_rho = atof(argv[i + 1]);
	if ((i = ArgPos((char *)"-threads", argc, argv)) > 0) num_threads = atoi(argv[i + 1]);
	total_samples *= 1000000;
	rho = init_rho;
	vertex = (struct ClassVertex *)calloc(max_num_vertices, sizeof(struct ClassVertex));
	TrainLINE();
	return 0;
}
