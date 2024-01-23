//   To compile: mpic++ matrix_multiplication.cpp -o matrix-multiplication
//   To run: mpirun -n 3 ./matrix-multiplication

#include <bits/stdc++.h>
#include <mpi.h>
using namespace std;

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    double startTime = MPI_Wtime();
    srand(time(0));

    int numOfMatrix = 9; // Total number of matrix
    int row1 = 4;        // Number of rows of the first matrix
    int col1 = 3;        // Number of cols of the first and rows of the 2nd matrix
    int col2 = 5;        // Number of cols of the 2nd matrix

    if (numOfMatrix % size != 0)
    {
        cout << "Number of total matrix should be Divisible by number of Process!!!\n";
        MPI_Finalize();
        return 0;
    }

    int matrix1[numOfMatrix][row1][col1];   // Array of the first matrix
    int matrix2[numOfMatrix][col1][col2];   // Array of the 2nd matrix
    int ansMatrix[numOfMatrix][row1][col2]; // Array of the ans matrix

    if (rank == 0) // Rank0 process will create the matrixes
    {
        for (int n = 0; n < numOfMatrix; n++)
        {
            for (int r = 0; r < row1; r++)
            {
                for (int c = 0; c < col1; c++)
                {
                    matrix1[n][r][c] = rand() % 10;
                }
            }

            for (int r = 0; r < col1; r++)
            {
                for (int c = 0; c < col2; c++)
                {
                    matrix2[n][r][c] = rand() % 10;
                }
            }
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);

    int subTotal = numOfMatrix / size;
    int mat1[subTotal][row1][col1];
    int mat2[subTotal][col1][col2];
    int mul[subTotal][row1][col2];

    MPI_Scatter(matrix1, subTotal * row1 * col1, MPI_INT, mat1, subTotal * row1 * col1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Scatter(matrix2, subTotal * col1 * col2, MPI_INT, mat2, subTotal * col1 * col2, MPI_INT, 0, MPI_COMM_WORLD);

    // Performing Multiplication
    for (int i = 0; i < subTotal; i++)
    {
        for (int j = 0; j < row1; j++)
        {
            for (int k = 0; k < col2; k++)
            {
                int sum = 0;
                for (int l = 0; l < col1; l++)
                {
                    sum += (mat1[i][j][l] * mat2[i][l][k]);
                }
                mul[i][j][k] = sum;
            }
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);

    MPI_Gather(mul, subTotal * row1 * col2, MPI_INT, ansMatrix, subTotal * row1 * col2, MPI_INT, 0, MPI_COMM_WORLD);

    if (rank == 0)
    {
        for (int n = 0; n < numOfMatrix; n++)
        {
            cout << "Result" << n << " =\n";
            for (int r = 0; r < row1; r++)
            {
                cout << "\t\t";
                for (int c = 0; c < col2; c++)
                {
                    cout << ansMatrix[n][r][c] << "  ";
                }
                cout << endl;
            }
        }
    }
    double endTime = MPI_Wtime();

    MPI_Barrier(MPI_COMM_WORLD);
    printf("Process %d took %f seconds.\n", rank, endTime - startTime);

    MPI_Finalize();
    return 0;
}