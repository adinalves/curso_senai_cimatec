#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <cuda.h>

#define BLOCK_SIZE 256
#define SOFTENING 1e-9f

typedef struct
{
    float x, y, z, vx, vy, vz, a, b;
} Body;

void randomizeBodies(float *data, int n)
{
    for (int i = 0; i < n; i++)
        data[i] = 2.0f * (rand() / (float)RAND_MAX) - 1.0f;
}

__global__ void bodyForce(Body *p, float dt, int n)
{

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;

    int stride = gridDim.x * blockDim.x;

    for (int i = idx; i < n; i += stride)
    {
        for (int k = idy; k < n; k += stride)
        {
            float Fx = 0.0f;
            float Fy = 0.0f;
            float Fz = 0.0f;
            for (int j = 0; j < n; j++)
            {
                float dx = p[j].x - p[i].x;
                float dy = p[j].y - p[i].y;
                float dz = p[j].z - p[i].z;
                float distSqr = dx * dx + dy * dy + dz * dz + SOFTENING;
                float invDist = rsqrtf(distSqr);
                float invDist3 = invDist * invDist * invDist;

                Fx += dx * invDist3;
                Fy += dy * invDist3;
                Fz += dz * invDist3;
            }
            p[i].vx += dt * Fx;
            p[i].vy += dt * Fy;
            p[i].vz += dt * Fz;
        }
    }
}

int main(const int argc, const char **argv)
{
    int nBodies = 30000; // size of the problem (bodies)

    if (argc > 1)
        nBodies = atoi(argv[1]);

    const float dt = 0.01f; // time step
    const int nIters = 10;  // simulation iterations
    int bytes = nBodies * sizeof(Body);
    float *buf = (float *)malloc(bytes);
    Body *p = (Body *)buf;

    randomizeBodies(buf, 6 * nBodies); // Init possible data

    /* device */
    float *d_buf;
    cudaMalloc(&d_buf, bytes);
    Body *d_p = (Body *)d_buf;

    int deviceId, numberOfSMs;
    cudaGetDevice(&deviceId);
    cudaDeviceGetAttribute(&numberOfSMs, cudaDevAttrMultiProcessorCount, deviceId);
    int NUMBER_OF_BLOCKS = numberOfSMs * 32;
    int NUMBER_OF_THREADS = 1024;

    const double t1 = omp_get_wtime();

    for (int iter = 1; iter <= nIters; iter++)
    {
        cudaMemcpy(d_buf, buf, bytes, cudaMemcpyHostToDevice);
        bodyForce<<<NUMBER_OF_BLOCKS, NUMBER_OF_THREADS>>>(d_p, dt, nBodies); // compute interbody forces
        cudaMemcpy(buf, d_buf, bytes, cudaMemcpyDeviceToHost);

        for (int i = 0; i < nBodies; i++)
        { // integrate position
            p[i].x += p[i].vx * dt;
            p[i].y += p[i].vy * dt;
            p[i].z += p[i].vz * dt;
        }
    }

    const double t2 = omp_get_wtime();

    double avgTime = (t2 - t1) / (double)(nIters - 1);

    float billionsOfOpsPerSecond = 1e-9 * nBodies * nBodies / avgTime;
    printf("\nSize (Bodies) = %d\n0", nBodies);
    printf("%0.3f Billion Interactions/second\n", billionsOfOpsPerSecond);
    printf("%0.3f second\n", avgTime);

    cudaFree(d_buf);
    free(buf);

    return 0;
}