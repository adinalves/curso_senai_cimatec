/*
@(#)Purpose:        Algebraic value applied in a function in C
@(#)Author:         Adine Alves
@(#)Usage:
@(*) Hotocompile:   mpicc algebraic_function.c -o algebraic_function
@(*) Hotoexecute:   mpirun -np 4 ./algebraic_function
*/

#include <stdio.h>
#include <mpi.h>
#define SIZE 4
int main(int argc, char **argv)
{
  double coef[4], x;
  double total = 0;
  int numberOfProcessors, id, to, tag = 1000;
  double result, value = 0;

  /*Iniatilize MPI*/
  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &numberOfProcessors);
  MPI_Comm_rank(MPI_COMM_WORLD, &id);
  MPI_Status status;

  switch (id)
  {
  case 0: /*Master*/
    coef[0] = 1;
    coef[1] = 2;
    coef[2] = 3;
    coef[3] = 4;
    x = 10;
    printf("\nf(x)=%.2lf*x^3+%.2lf*x^2+%.2lf*x+%.2lf\n", coef[0], coef[1], coef[2], coef[3]);
    printf("\n");

    for (to = 1; to < numberOfProcessors; to++)
    {
      if (to == 3)
      {
        MPI_Send(&coef, SIZE, MPI_DOUBLE, to, tag, MPI_COMM_WORLD);
        MPI_Send(&x, 1, MPI_DOUBLE, to, tag, MPI_COMM_WORLD);
      }
      else
      {
        MPI_Send(&coef[to - 1], 1, MPI_DOUBLE, to, tag, MPI_COMM_WORLD);
        MPI_Send(&x, 1, MPI_DOUBLE, to, tag, MPI_COMM_WORLD);
      }
    }

    for (to = 1; to < numberOfProcessors; to++)
    {
      MPI_Recv(&result, 1, MPI_DOUBLE, to, tag, MPI_COMM_WORLD, &status);
      total = total + result;
    }
    printf("\nf(%.2lf) = %.2lf*x^3 + %.2lf*x^2 + %.2lf*x + %.2lf = %.2lf\n", x, coef[0], coef[1], coef[2], coef[3], total);
    printf("total = %.2lf\n", total);

    break;

  case 1: /*Slave 1*/
    MPI_Recv(&coef, 1, MPI_DOUBLE, 0, tag, MPI_COMM_WORLD, &status);
    MPI_Recv(&x, 1, MPI_DOUBLE, 0, tag, MPI_COMM_WORLD, &status);
    value = coef[0] * x * x * x;
    MPI_Send(&value, 1, MPI_DOUBLE, 0, tag, MPI_COMM_WORLD);
    break;

  case 2: /*Slave 2*/
    MPI_Recv(&coef, 1, MPI_DOUBLE, 0, tag, MPI_COMM_WORLD, &status);
    MPI_Recv(&x, 1, MPI_DOUBLE, 0, tag, MPI_COMM_WORLD, &status);
    value = coef[0] * x * x;
    MPI_Send(&value, 1, MPI_DOUBLE, 0, tag, MPI_COMM_WORLD);
    break;

  case 3: /*Slave 3*/
    MPI_Recv(&coef, SIZE, MPI_DOUBLE, 0, tag, MPI_COMM_WORLD, &status);
    MPI_Recv(&x, 1, MPI_DOUBLE, 0, tag, MPI_COMM_WORLD, &status);
    value = coef[2] * x + coef[3];
    MPI_Send(&value, 1, MPI_DOUBLE, 0, tag, MPI_COMM_WORLD);
    break;
  }

  MPI_Finalize();
  return 0;
}