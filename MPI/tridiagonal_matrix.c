#include <stdio.h>
#include <mpi.h>
#define ORDER 4

void printMatrix(int m[][ORDER])
{
    int i, j;
    for (i = 0; i < ORDER; i++)
    {
        printf("| ");
        for (j = 0; j < ORDER; j++)
        {
            printf("%3d ", m[i][j]);
        }
        printf("|\n");
    }
    printf("\n");
}
int main(int argc, char **argv)
{

    int numberOfProcessors, id, to, tag = 1000;
    int result[ORDER];
    int value[ORDER];

    /*Iniatilize MPI*/
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &numberOfProcessors);
    MPI_Comm_rank(MPI_COMM_WORLD, &id);
    MPI_Status status;


    switch (id)
    {
    case 0:  // Master
        int k[3] = {100, 200, 300};
        int matrix[ORDER][ORDER], i, j;
        int matrix_1[ORDER];
        int matrix_2[ORDER];
        for (i = 0; i < ORDER; i++)
        {
            for (j = 0; j < ORDER; j++)
            {
                if (i == j)
                {
                    matrix[i][j] = i + j + 1;
                    matrix_1[i] = matrix[i][j];
                }

                else if (i == (j + 1))
                {
                    matrix[i][j] = i + j + 1;
                    matrix[j][i] = matrix[i][j];
                    matrix_2[j] = matrix[i][j];
                }
                else
                    matrix[i][j] = 0;
            }
        }

        for (to = 1; to < numberOfProcessors; to++)
        {
            if (to == 1)
            {
                MPI_Send(&matrix_1, ORDER, MPI_INT, to, tag, MPI_COMM_WORLD);
                MPI_Send(&k[to-1], 1, MPI_INT, to, tag, MPI_COMM_WORLD);
            }

            else if (to == 2)
            {
                MPI_Send(&matrix_2, ORDER, MPI_INT, to, tag, MPI_COMM_WORLD);
                 MPI_Send(&k[to-1], 1, MPI_INT, to, tag, MPI_COMM_WORLD);
            }

            else if (to == 3)
            {
                MPI_Send(&matrix_2, ORDER, MPI_INT, to, tag, MPI_COMM_WORLD);
                 MPI_Send(&k[to-1], 1, MPI_INT, to, tag, MPI_COMM_WORLD);
            }
        }
        for (to = 1; to < numberOfProcessors; to++)
        {
            if (to == 1)
            {
                MPI_Recv(&result, ORDER, MPI_INT, to, tag, MPI_COMM_WORLD, &status);
                for (i = 0; i < ORDER; i++)
                {
                    matrix[i][i] = result[i]; // main diagonal
                }
            }

            else if (to == 2)
            {
                MPI_Recv(&result, ORDER, MPI_INT, to, tag, MPI_COMM_WORLD, &status);
                for (i = 0; i < ORDER; i++)
                {
                    matrix[i + 1][i] = result[i]; // subdiagonal
                }
            }

            else if (to == 3)
            {
                MPI_Recv(&result, ORDER, MPI_INT, to, tag, MPI_COMM_WORLD, &status);
                for (i = 0; i < ORDER; i++)
                {
                    matrix[i][i + 1] = result[i]; // superdiagonal
                }
            }
        }
        printMatrix(matrix);
        break;

    case 1: /*Slave 1*/
        MPI_Recv(&matrix_1, ORDER, MPI_INT, 0, tag, MPI_COMM_WORLD, &status);
        MPI_Recv(&k, 1, MPI_INT, 0, tag, MPI_COMM_WORLD, &status);
        for (i = 0; i < ORDER; i++)
        {
            value[i] = matrix_1[i] + k[0]; // main diagonal
        }
        MPI_Send(&value, ORDER, MPI_INT, 0, tag, MPI_COMM_WORLD);
        break;

    case 2: /*Slave 2*/
        MPI_Recv(&matrix_2, ORDER, MPI_INT, 0, tag, MPI_COMM_WORLD, &status);
         MPI_Recv(&k, 1, MPI_INT, 0, tag, MPI_COMM_WORLD, &status);
        for (i = 0; i < ORDER; i++)
        {
            value[i] = matrix_2[i] + k[0]; // subdiagonal
        }
        MPI_Send(&value, ORDER, MPI_INT, 0, tag, MPI_COMM_WORLD);
        break;

    case 3: /*Slave 3*/
        MPI_Recv(&matrix_2, ORDER, MPI_INT, 0, tag, MPI_COMM_WORLD, &status);
         MPI_Recv(&k, 1, MPI_INT, 0, tag, MPI_COMM_WORLD, &status);
        for (i = 0; i < ORDER; i++)
        {
            value[i] = matrix_2[i] + k[0]; // superdiagonal
        }
        MPI_Send(&value, ORDER, MPI_INT, 0, tag, MPI_COMM_WORLD);
        break;
    }

    MPI_Finalize();
    return 0;
}