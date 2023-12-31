#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

#define BLOCK_SIZE 256
#define SOFTENING 1e-9f

typedef struct
{
    float x, y, z, vx, vy, vz;
} Body;

void randomizeBodies(float *data, int n)
{
    for (int i = 0; i < n; i++)
        data[i] = 2.0f * (rand() / (float)RAND_MAX) - 1.0f;
}

void bodyForce(Body *p, float dt, int n)
{
#pragma omp parallel for
    for (int i = 0; i < n; ++i)
    {
        float Fx = 0.0f;
        float Fy = 0.0f;
        float Fz = 0.0f;

#pragma omp parallel for reduction(+ : Fx, Fy, Fz)
        for (int j = 0; j < n; j++)
        {
            float dx = p[j].x - p[i].x;
            float dy = p[j].y - p[i].y;
            float dz = p[j].z - p[i].z;
            float distSqr = dx * dx + dy * dy + dz * dz + SOFTENING;
            float invDist = 1/sqrt(distSqr);
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

    randomizeBodies(buf, 6 * nBodies); // Init pos/vel data

    const double t1 = omp_get_wtime();

    for (int iter = 1; iter <= nIters; iter++)
    {
        bodyForce(p, dt, nBodies); // compute interbody forces

#pragma omp parallel for
        for (int i = 0; i < nBodies; i++)
        {
            p[i].x += p[i].vx * dt;
            p[i].y += p[i].vy * dt;
            p[i].z += p[i].vz * dt;
        }
    }

    printf("x: %f\n", p->x);
    printf("y: %f\n", p->y);
    printf("z: %f\n", p->z);
    printf("vx: %f\n", p->vx);
    printf("vy: %f\n", p->vy);
    printf("vz: %f\n", p->vz);

    const double t2 = omp_get_wtime();

    double avgTime = (t2 - t1) / (double)(nIters - 1);

    float billionsOfOpsPerSecond = 1e-9 * nBodies * nBodies / avgTime;
    printf("\nSize (Bodies) = %d\n", nBodies);
    printf("%0.3f Billion Interactions/second\n", billionsOfOpsPerSecond);
    printf("%0.3f second\n", avgTime);

    free(buf);

    return 0;
}
