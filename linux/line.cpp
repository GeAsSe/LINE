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

const int hash_table_size = 30000000;  // 30 million  3千万
const int neg_table_size = 1e8; // 100 million 1亿
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
// num_edges: 记录边的数量
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
// 读取网络数据，初始化边相关的三个向量, 初始化vertex_hash_table和vertex数组
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

	//初始化边相关的三个向量，设置向量大小与边的数量相对应
	edge_source_id = (int *)malloc(num_edges*sizeof(int));
	edge_target_id = (int *)malloc(num_edges*sizeof(int));
	edge_weight = (double *)malloc(num_edges*sizeof(double));
	if (edge_source_id == NULL || edge_target_id == NULL || edge_weight == NULL)
	{
		printf("Error: memory allocation failed!\n");
		exit(1);
	}

	//2. 第二次读二进制文件network_file，对 每条边、每个顶点 的数据结构进行初始化
	fin = fopen(network_file, "rb");
	num_vertices = 0; //初始化顶点的数量为0

	//根据边数，循环读行
	for (int k = 0; k != num_edges; k++)
	{
		fscanf(fin, "%s %s %lf", name_v1, name_v2, &weight);  //读取一行，将对应值保存到局部变量中

		// 每读10000行，更新一次精确到三位小数的当前处理进度（百分数）
		if (k % 10000 == 0){
			printf("Reading edges: %.3lf%%%c", k / (double)(num_edges + 1) * 100, 13);
			fflush(stdout);
		}

		// 检查源节点是否曾加入到 *vertex数组 中
		vid = SearchHashTable(name_v1);
		// 若否则将其加入
		if (vid == -1){
			vid = AddVertex(name_v1);
		}
		// 更新 *vertex数组 中此节点的度，加上这条边的权重
		vertex[vid].degree += weight;
		// 令 此边的源节点 指向对应 *vertex数组 中的下标
		edge_source_id[k] = vid;

		// 检查目标节点是否曾加入到 *vertex数组 中
		vid = SearchHashTable(name_v2);
		if (vid == -1){
			vid = AddVertex(name_v2);
		}
		// 更新 *vertex数组 中此节点的度，加上这条边的权重
		vertex[vid].degree += weight;
		// 令 此边的目标节点 指向对应 *vertex数组 中的下标
		edge_target_id[k] = vid;

		//设置 此边的权重
		edge_weight[k] = weight;
	}
	fclose(fin);
	printf("Number of vertices: %d          \n", num_vertices);
}

// alias sampling 是一种高效的抽样算法，使用O(n)或O(nlogn)的时间进行预处理，之后每次采样只要O(1)的时间
// alias sampling 方法的预处理需要构造两个数组：alias和prob
// alias数组的长度为num_edges，每个元素是一个整数，用来存放结果，即采样边的下标
// prob数组的长度为num_edges，每个元素是一个浮点数，用来存放采样概率：
// 		若随机数rand_value2小于等于prob[k]，则采样边的下标为k
// 		若随机数rand_value2大于prob[k]，则采样边的下标为alias[k]
// InitAliasTable函数进行了这个预处理操作，构造了alias和prob数组
/* The alias sampling algorithm, which is used to sample an edge in O(1) time. */
void InitAliasTable()
{
	alias = (long long *)malloc(num_edges*sizeof(long long));  //alias是一个long long数组
	prob = (double *)malloc(num_edges*sizeof(double));  //prob是一个double数组
	if (alias == NULL || prob == NULL)
	{
		printf("Error: memory allocation failed!\n");
		exit(1);
	}

	double *norm_prob = (double*)malloc(num_edges*sizeof(double));  //norm_prob是选择每条边的norm概率数组， norm_prob[k] = edge_weight[k] / sum(edge_weight) * num_edges
	long long *large_block = (long long*)malloc(num_edges*sizeof(long long));  //large_block数组，用于存放norm概率大于1的边的下标
	long long *small_block = (long long*)malloc(num_edges*sizeof(long long));  //small_block数组，用于存放norm概率小于1的边的下标
	if (norm_prob == NULL || large_block == NULL || small_block == NULL)
	{
		printf("Error: memory allocation failed!\n");
		exit(1);
	}

	double sum = 0;  //sum为所有边的权重之和
	long long cur_small_block, cur_large_block;  //这两个指针用于遍历small_block和large_block数组
	long long num_small_block = 0, num_large_block = 0;  //这两个指针用于初始化small_block和large_block数组，并记录数组大小

	for (long long k = 0; k != num_edges; k++) sum += edge_weight[k];  //计算所有边的权重之和 O(n)
	for (long long k = 0; k != num_edges; k++) norm_prob[k] = edge_weight[k] * num_edges / sum;  //计算每条边的norm概率并赋值给norm_prob数组 O(n)

	//将norm概率大于等于1的边的下标存放到large_block数组中，将norm概率小于1的边的下标存放到small_block数组中
	for (long long k = num_edges - 1; k >= 0; k--)
	{
		if (norm_prob[k]<1)
			small_block[num_small_block++] = k;
		else
			large_block[num_large_block++] = k;
	}

	//若num_small_block或num_large_block不为0，则进行以下操作
	while (num_small_block && num_large_block)
	{
		//倒过来遍历small_block和large_block数组
		cur_small_block = small_block[--num_small_block];
		cur_large_block = large_block[--num_large_block];

		//cur_small_block是边的下标，其norm_prob概率即为prob(阈值概率)
		prob[cur_small_block] = norm_prob[cur_small_block];
		//其alias即为对应cur_large_block(也是一个边的下标)
		alias[cur_small_block] = cur_large_block;
		// （**关键一步）从cur_large_block的norm_prob中挖去一部分，来补全cur_small_block的norm_prob，使其为1
		norm_prob[cur_large_block] = norm_prob[cur_large_block] + norm_prob[cur_small_block] - 1;
		//在这之后，cur_large_block的norm_prob可能大于1，也可能小于1，所以要重新放回到large_block或small_block中
		if (norm_prob[cur_large_block] < 1)
			small_block[num_small_block++] = cur_large_block;
		else
			large_block[num_large_block++] = cur_large_block;
	}

	//浮点数计算可能有误差，如果有剩余的prob没有处理，则直接赋值为1
	while (num_large_block) prob[large_block[--num_large_block]] = 1;
	while (num_small_block) prob[small_block[--num_small_block]] = 1;

	free(norm_prob);
	free(small_block);
	free(large_block);
}

//用两个[0,1]之间的随机数进行采样，第一个决定下标，第二个根据prob[k]决定是k还是alias[k]
long long SampleAnEdge(double rand_value1, double rand_value2)
{
	long long k = (long long)num_edges * rand_value1;
	return rand_value2 < prob[k] ? k : alias[k];
}

/* Initialize the vertex embedding and the context embedding */
// 根据顶点数量num_vertices和嵌入向量维度dim，初始化顶点嵌入向量和上下文嵌入向量
void InitVector()
{
	long long a, b;

	//对嵌入向量进行初始化
	a = posix_memalign((void **)&emb_vertex, 128, (long long)num_vertices * dim * sizeof(real));
	if (emb_vertex == NULL) { printf("Error: memory allocation failed\n"); exit(1); }
	//二重循环，对每个顶点的每个维度，随机初始化一个[-0.5/dim, 0.5/dim]之间的值
	//emb_vertex是一个二维数组，其构造图示意如下：
	/*
	[
	vertex1[d1, d2, ..., dn]
	vertex2[d1, d2, ..., dn]
	...
	vertexm[d1, d2, ..., dn]
	]
	*/
	for (b = 0; b < dim; b++){
		for (a = 0; a < num_vertices; a++){
			emb_vertex[a * dim + b] = (rand() / (real)RAND_MAX - 0.5) / dim;
		}
	}

	//对嵌入上下文向量进行初始化，流程同上，只是emb_context初始化所有位置为0
	a = posix_memalign((void **)&emb_context, 128, (long long)num_vertices * dim * sizeof(real));
	if (emb_context == NULL) { printf("Error: memory allocation failed\n"); exit(1); }
	for (b = 0; b < dim; b++) for (a = 0; a < num_vertices; a++)
		emb_context[a * dim + b] = 0;
}

// Negative Table用于负采样，其长度为1亿，每个位置存放一个顶点的下标
// 对于任意顶点u，其被采样的概率正比于 degree(u)^NEG_SAMPLING_POWER
// 这个neg_table把采样的概率处理成1亿个等长的区间，每个顶点对应的区间的数正比于其被采样的概率。
/* Sample negative vertex samples according to vertex degrees */
void InitNegTable()
{
	double sum = 0, cur_sum = 0, por = 0;//portion
	int vid = 0;
	neg_table = (int *)malloc(neg_table_size * sizeof(int)); //neg_table是一个int数组，长度为1亿
	//计算所有顶点的 degree^NEG_SAMPLING_POWER 之和，计为sum
	for (int k = 0; k != num_vertices; k++) sum += pow(vertex[k].degree, NEG_SAMPLING_POWER);
	//构造neg_table, 
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
// 构造[-SIGMOID_BOUND, SIGMOID_BOUND]之间的1000个等长区间，每个区间的sigmoid值存放在sigmoid_table中
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

//带分段的sigmoid函数：
/*
f(x) = 1, x > 6
f(x) = 1 / (1 + exp(-x)), -6 <= x <= 6, 这一步直接查表计算
f(x) = 0, x < -6
*/
real FastSigmoid(real x)
{
	if (x > SIGMOID_BOUND) return 1;
	else if (x < -SIGMOID_BOUND) return 0;
	int k = (x + SIGMOID_BOUND) * sigmoid_table_size / SIGMOID_BOUND / 2;
	return sigmoid_table[k];
}

/* Fastly generate a random integer */
//这个Rand函数用于对负采样表产生[0, neg_table_size)之间的随机数
int Rand(unsigned long long &seed)
{
	seed = seed * 25214903917 + 11;
	return (seed >> 16) % neg_table_size;
}

/* Update embeddings */
// 使用AliasMethod进行边的采样，得到源节点 u，目标节点 v；
// vec_u 为源节点的embedding，vec_v 为目标节点的embedding
// 
void Update(real *vec_u, real *vec_v, real *vec_error, int label)
{
	real x = 0, g;
	for (int c = 0; c != dim; c++){
		x += vec_u[c] * vec_v[c];
	}
	g = (label - FastSigmoid(x)) * rho;
	for (int c = 0; c != dim; c++){
		vec_error[c] += g * vec_v[c];
	}
	for (int c = 0; c != dim; c++){
		vec_v[c] += g * vec_u[c];
	}
}

void *TrainLINEThread(void *id) {
	// u是源节点在vertex列表中的下标，v是目标节点在vertex列表中的下标，lu是源节点的embedding在emb_vertex中的起始位置，lv是目标节点的embedding在emb_vertex中的起始位置
	long long u, v, lu, lv, target, label;
	long long count = 0, last_count = 0, curedge;
	unsigned long long seed = (long long)id;
	real *vec_error = (real *)calloc(dim, sizeof(real));

	while (1)
	{
		//judge for exit
		if (count > total_samples / num_threads + 2) break;

		//每处理10000条边，更新一次学习率rho，更新一次当前处理进度
		if (count - last_count > 10000)
		{
			current_sample_count += count - last_count;
			last_count = count;
			printf("%cRho: %f  Progress: %.3lf%%", 13, rho, (real)current_sample_count / (real)(total_samples + 1) * 100);
			fflush(stdout);
			// 令学习率rho随着当前处理进度的增加而减小
			rho = init_rho * (1 - current_sample_count / (real)(total_samples + 1));
			if (rho < init_rho * 0.0001) rho = init_rho * 0.0001;
		}

		// [正采样] 随机采样一条边，得到源节点u，目标节点v
		curedge = SampleAnEdge(gsl_rng_uniform(gsl_r), gsl_rng_uniform(gsl_r));
		u = edge_source_id[curedge];
		v = edge_target_id[curedge];

		lu = u * dim;
		for (int c = 0; c != dim; c++){
			vec_error[c] = 0;
		}

		// [负采样] NEGATIVE SAMPLING
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
		for (int c = 0; c != dim; c++) {
			emb_vertex[c + lu] += vec_error[c];
		}

		count++;
	}
	free(vec_error);
	pthread_exit(NULL);
}

//Output函数定义了embedding文件的格式
void Output()
{
	FILE *fo = fopen(embedding_file, "wb");
	fprintf(fo, "%d %d\n", num_vertices, dim);  //第一行记录嵌入矩阵的行数和列数
	for (int a = 0; a < num_vertices; a++)  //之后每一行记录一个顶点的嵌入向量
	{
		fprintf(fo, "%s ", vertex[a].name); //第一个元素是顶点的名字
		if (is_binary) for (int b = 0; b < dim; b++) fwrite(&emb_vertex[a * dim + b], sizeof(real), 1, fo); //如果是二进制文件，直接写入
		else for (int b = 0; b < dim; b++) fprintf(fo, "%lf ", emb_vertex[a * dim + b]); //如果是文本文件，写入时加空格
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

	//创建多个线程来进行TrainLINEThread工作；
	for (a = 0; a < num_threads; a++){
		pthread_create(&pt[a], NULL, TrainLINEThread, (void *)a);
	}
	for (a = 0; a < num_threads; a++){
		pthread_join(pt[a], NULL);
	}
	printf("\n");
	clock_t finish = clock();
	printf("Total time: %lf\n", (double)(finish - start) / CLOCKS_PER_SEC);

	Output();
}

//给定要查找的参数选项和参数列表，如果找到了对应的参数选项，返回其在参数列表中的下标，否则返回-1
int ArgPos(char *str, int argc, char **argv) {
	int a;
	for (a = 1; a < argc; a++){
		if (!strcmp(str, argv[a])) {
			if (a == argc - 1) {
				printf("Argument missing for %s\n", str);
				exit(1);
			}
			return a;
		}
	}
	return -1;
}

// ./line -train net.txt -output vec.txt -binary 1 -size 200 -order 2 -negative 5 -samples 100 -rho 0.025 -threads 20
int main(int argc, char **argv) {
	int i;
	//如果没有带参数运行命令，打印帮助信息
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
	total_samples *= 1000000; //samples选项的单位是百万
	rho = init_rho;
	vertex = (struct ClassVertex *)calloc(max_num_vertices, sizeof(struct ClassVertex));
	TrainLINE();
	return 0;
}
