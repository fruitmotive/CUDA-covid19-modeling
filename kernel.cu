#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <conio.h>
#include <curand.h>
#include <curand_kernel.h>

#define T 150 // время моделирования, какждая единица - один день 
#define N 20480 // кол-во точек (людей) в данной системе (городе)
#define RX 700 // размер коробки (города) по оси X
#define RY 700 // размер коробки (города) по оси Y
#define NOI 1 // кол-во изначально зараженных
#define MAX_R 35 // максимальное перемещение точки (человека) по оси за день
#define INF_R 3 // радиус заражения
#define T_REC 20 // время выздоровления   


// перемещение точек (людей) внутри системы (города) (CPU)
void coordinates_CPU (float *x, float *y, int i, int nt)
{
	x[nt * N + i] = x[(nt-1) * N + i] + 2 * MAX_R * ((float)rand() / RAND_MAX) - MAX_R;
	y[nt * N + i] = y[(nt-1) * N + i] + 2 * MAX_R * ((float)rand() / RAND_MAX) - MAX_R;
	while ((fabsf(x[nt * N + i]) > RX/2) || (fabsf(y[nt * N + i]) > RY/2))
	{
		x[nt * N + i] = x[(nt - 1) * N + i] + 2 * MAX_R * ((float)rand() / RAND_MAX) - MAX_R;
		y[nt * N + i] = y[(nt - 1) * N + i] + 2 * MAX_R * ((float)rand() / RAND_MAX) - MAX_R;
	}
}


// кто заболеет? (CPU)
void status_CPU (float *x, float *y, int *s, int i, int nt)
{
	float xx, yy, rr;
	for (int j = 0; j < N; j++)
	{
		if ((j != i) && (s[i] == 0) && (s[j] == 1))
		{  
			xx = x[nt * N + i] - x[nt * N + j];
			yy = y[nt * N + i] - y[nt * N + j];
			rr = sqrtf(xx * xx + yy * yy);
			if (rr < INF_R)
			{
				s[i] = 1;
			}
		}
	}
}


// контроль (CPU)
void control_CPU(int *s, int *t, int *q, int nt, int i)
{
	if (s[i] == 1)
	{
		t[i] = t[i] + 1;
	}
	if (t[i] == T_REC)
	{
		s[i] = 2;
	}
	if (s[i] == 1)
	{
		q[nt] = q[nt] + s[i];
	}
}


// кто заболеет? (GPU)
__global__ void status_GPU(float *x, float *y, int *s, int nt)
{
	float xx, yy, rr;
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	for (int j = 0; j < N; j++)
	{
		if ((j != i) && (s[i] == 0) && (s[j] == 1))
		{
			xx = x[nt * N + i] - x[nt * N + j];
			yy = y[nt * N + i] - y[nt * N + j];
			rr = sqrtf(xx * xx + yy * yy);
			if (rr < INF_R)
			{
				s[i] = 1;
			}
		}
	}
}


// контроль (GPU)
__global__ void control_GPU(int *s, int *t, int nt)
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	if (t[i] == T_REC)
	{
		s[i] = 2;
	}
	if (s[i] == 1)
	{
		t[i] = t[i] + 1;
	}
}


void control_2_CPU(int *s, int *q, int i, int nt)
{
	if (s[i] == 1)
	{
		q[nt] = q[nt] + s[i];
	}
}

int main()
{
	FILE *CPU_coordinates;
	FILE *CPU_quantity;
	FILE *CPU_status;


	FILE *GPU_coordinates;
	FILE *GPU_quantity;
	FILE *GPU_status;


	float *x_CPU, *y_CPU;
	int *q_CPU, *s_CPU, *t_CPU;
	x_CPU = (float*)malloc(sizeof(float)*(T + 1)*N); 
	y_CPU = (float*)malloc(sizeof(float)*(T + 1)*N);
	s_CPU = (int*)malloc(sizeof(int)*N);
	q_CPU = (int*)malloc(sizeof(int)*(T + 1));
	t_CPU = (int*)malloc(sizeof(int)*N);
	time_t start_CPU, stop_CPU;


	float *x_GPU, *y_GPU;
	int *s_GPU, *t_GPU;
	cudaMalloc((void**)&x_GPU, sizeof(float)*(T + 1)*N);
	cudaMalloc((void**)&y_GPU, sizeof(float)*(T + 1)*N);
	cudaMalloc((void**)&s_GPU, sizeof(int)*N);
	cudaMalloc((void**)&t_GPU, sizeof(int)*N);
	cudaEvent_t start_GPU, stop_GPU;
	cudaEventCreate(&start_GPU);
	cudaEventCreate(&stop_GPU);
	float timerValueGPU;
	int N_thread = 256; int N_block = N / N_thread;


	float *x_for_GPU, *y_for_GPU;
	int *q_for_GPU, *s_for_GPU;

	//x_for_GPU = (float*)malloc(sizeof(float)*(T + 1)*N);
	//y_for_GPU = (float*)malloc(sizeof(float)*(T + 1)*N);
	//s_for_GPU = (int*)malloc(sizeof(int)*N);
	//q_for_GPU = (int*)malloc(sizeof(int)*(T + 1));

	cudaHostAlloc((void**)&x_for_GPU, sizeof(float)*(T + 1)*N, cudaHostAllocDefault);
	cudaHostAlloc((void**)&y_for_GPU, sizeof(float)*(T + 1)*N, cudaHostAllocDefault);
	cudaHostAlloc((void**)&s_for_GPU, sizeof(int)*N, cudaHostAllocDefault);
	cudaHostAlloc((void**)&q_for_GPU, sizeof(int)*(T + 1), cudaHostAllocDefault);


	srand(time(NULL));
	for (int i = 0; i < NOI; i++)
	{
		t_CPU[i] = 1;
	}

	for (int i = NOI; i < N; i++)
	{
		t_CPU[i] = 0;
	}


	//определяем начальные координаты
	for (int i = 0; i < N; i++)
	{
	    x_CPU[i] = RX * ((float)rand() / RAND_MAX) - RX / 2;
		y_CPU[i] = RY * ((float)rand() / RAND_MAX) - RY / 2;
		x_for_GPU[i] = x_CPU[i];
		y_for_GPU[i] = y_CPU[i];
    }

	//определяем начально зараженных
	for (int i = 0; i < NOI; i++)
	{
		s_CPU[i] = 1;
	}
	
	
	//определяем начально здоровых
	for (int i = NOI; i < N; i++)
	{
		s_CPU[i] = 0;
	}
	

	//записываем информацию о заболевших изначально
	q_CPU[0] = NOI;
	q_for_GPU[0] = NOI;
	for (int i = 1; i < T + 1; i++)
	{
		q_CPU[i] = 0;
		q_for_GPU[i] = 0;
	}


	//CUDA вариант
	cudaEventRecord(start_GPU, 0);

	cudaMemcpy(s_GPU, s_CPU, sizeof(int)*N, cudaMemcpyHostToDevice);
	cudaMemcpy(t_GPU, t_CPU, sizeof(int)*N, cudaMemcpyHostToDevice);

	for (int nt = 1; nt < T + 1; nt++)
	{
		for (int i = 0; i < N; i++)
		{
			coordinates_CPU(x_for_GPU, y_for_GPU, i, nt);
		}
	}
	cudaMemcpy(x_GPU, x_for_GPU, sizeof(float)*(T + 1)*N, cudaMemcpyHostToDevice);
	cudaMemcpy(y_GPU, y_for_GPU, sizeof(float)*(T + 1)*N, cudaMemcpyHostToDevice);
	for (int nt = 1; nt < T + 1; nt++)
	{
		status_GPU << < N_block, N_thread >> > (x_GPU, y_GPU, s_GPU, nt);
		control_GPU << < N_block, N_thread >> > (s_GPU, t_GPU, nt);
	    cudaMemcpy(s_for_GPU, s_GPU, sizeof(int)*N, cudaMemcpyDeviceToHost);
		for (int i = 0; i < N; i++)
		{
			control_2_CPU(s_for_GPU, q_for_GPU, i, nt);
		}
	}

	
	cudaEventRecord(stop_GPU, 0);
	cudaEventSynchronize(stop_GPU);
	cudaEventElapsedTime(&timerValueGPU, start_GPU, stop_GPU);
	printf("GPU calculation time: %f msec\n", timerValueGPU);


	GPU_coordinates = fopen("coordinates_GPU.txt", "w");
	for (int nt = 0; nt < T + 1; nt++)
	{
		for (int i = 0; i < N; i++)
		{
			fprintf(GPU_coordinates, "%f %f\n", x_for_GPU[nt * N + i], y_for_GPU[nt * N + i]);
		}
	}
	fclose(GPU_coordinates);

	GPU_quantity = fopen("quantity_GPU.txt", "w");
	for (int nt = 0; nt < T + 1; nt++)
	{
		fprintf(GPU_quantity, "%d %d\n", nt, q_for_GPU[nt]);
	}
	fclose(GPU_quantity);

	GPU_status = fopen("status_GPU.txt", "w");
	for (int i = 0; i < N; i++)
	{
		fprintf(GPU_status, "%d %d\n", i, s_for_GPU[i]);
	}
	fclose(GPU_status);





	//CPU вариант
    start_CPU = clock();
	for (int nt = 1; nt < T + 1; nt++)
	{
		for (int i = 0; i < N; i++)
		{
			coordinates_CPU(x_CPU, y_CPU, i, nt);
		}
		for (int i = 0; i < N; i++)
		{
			status_CPU(x_CPU, y_CPU, s_CPU, i, nt);
		}
		for (int i = 0; i < N; i++)
		{
			control_CPU(s_CPU, t_CPU, q_CPU, nt, i);
		}
	}
	stop_CPU = clock();
	printf("CPU calculation time: %f seconds", (double)(stop_CPU - start_CPU) / CLOCKS_PER_SEC);


	CPU_coordinates = fopen("coordinates_CPU.txt", "w");
	for (int nt = 0; nt < T + 1; nt++)
	{
		for (int i = 0; i < N; i++)
		{
			fprintf(CPU_coordinates, "%f %f\n", x_CPU[nt * N + i], y_CPU[nt * N + i]);
        }
	}
	fclose(CPU_coordinates);

	CPU_quantity = fopen("quantity_CPU.txt", "w");
	for (int nt = 0; nt < T + 1; nt++)
	{
		fprintf(CPU_quantity, "%d %d\n", nt, q_CPU[nt]);
	}
	fclose(CPU_quantity);

	CPU_status = fopen("status_CPU.txt", "w");
	for (int i = 0; i < N; i++)
	{
		fprintf(CPU_status, "%d %d\n", i, s_CPU[i]);
	}
	fclose(CPU_status);


    free(x_CPU);
	free(y_CPU);
	free(s_CPU);
	free(q_CPU);
	free(t_CPU);
	//free(x_for_GPU);
	//free(y_for_GPU);
	//free(s_for_GPU);
	//free(q_for_GPU);
	cudaFreeHost(x_for_GPU);
	cudaFreeHost(y_for_GPU);
	cudaFreeHost(s_for_GPU);
	cudaFreeHost(q_for_GPU);
	cudaFree(x_GPU);
	cudaFree(y_GPU);
	cudaFree(s_GPU);
	cudaFree(t_GPU);
	cudaEventDestroy(start_GPU);
	cudaEventDestroy(stop_GPU);
	getch();
	return 0;
}




	



















	

	

