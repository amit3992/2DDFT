// Distributed two-dimensional Discrete FFT transform
// AMIT KULKARNI
// GT ID: 903038158
// ECE6122 Project 1 //

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <signal.h>
#include <math.h>
#include <mpi.h>

#include "Complex.h"
#include "InputImage.h"

using namespace std;

void Transform2D(const char* inputFN)
{ // Do the 2D transform here.
  // 1) Use the InputImage object to read in the Tower.txt file and
  //    find the width/height of the input image.
  // 2) Use MPI to find how many CPUs in total, and which one
  //    this process is
  // 3) Allocate an array of Complex object of sufficient size to
  //    hold the 2d DFT results (size is width * height)
  // 4) Obtain a pointer to the Complex 1d array of input data
  // 5) Do the individual 1D transforms on the rows assigned to your CPU
  // 6) Send the resultant transformed values to the appropriate
  //    other processors for the next phase.
  // 6a) To send and receive columns, you might need a separate
  //     Complex array of the correct size.
  // 7) Receive messages from other processes to collect your columns
  // 8) When all columns received, do the 1D transforms on the columns
  // 9) Send final answers to CPU 0 (unless you are CPU 0)
  //   9a) If you are CPU 0, collect all values from other processors
  //       and print out with SaveImageData().
  // Step (1) in the comments is the line above.
  // Your code here, steps 2-9
}

void Transform1D(Complex* h, int w, Complex* H)
{
  // Implement a simple 1-d DFT using the double summation equation
  // given in the assignment handout.  h is the time-domain input
  // data, w is the width (N), and H is the output array.
  //Complex D;
  Complex val = 0;
  double theta = 0.0;
  for (int n=0; n < w; n++)
  {
    Complex D = 0;
    for(int k=0; k < w; k++)
    {
      theta = ((2*M_PI*n*k)/w);
      val = Complex(cos(theta), -1*sin(theta));
      D = D + (h[k]*val);
    }
      H[n] = D;
  }

}

void InverseTransform1D(Complex* h, int w, Complex* H)
{
  // Implement a simple 1-d DFT using the double summation equation
  // given in the assignment handout.  h is the time-domain input
  // data, w is the width (N), and H is the output array.
  //Complex D;
  Complex val = 0;
  double theta = 0.0;
  for (int n = 0; n < w; n++)
  {
    Complex D = 0;
    for(int k = 0; k < w; k++)
    {
      theta = ((2*M_PI*n*k)/w);
      val = Complex(cos(theta), sin(theta));
      D = D + (h[k]*val);
    }
      H[n].real = D.real/w;
      H[n].imag = D.imag/w;
  }

}

int main(int argc, char** argv)
{
  string fn("Tower.txt"); // default file name
  if (argc > 1) fn = string(argv[1]);  // if name specified on cmd line
  InputImage image(fn.c_str());  // Create the helper object for reading the image

  // Declaring local variables required for computation.
  int numtasks, rank, rc;
  int ht,wd = 0;
  int rowsPerCPU = 0;
  wd = image.GetWidth();
  ht = image.GetHeight();

  // Declare complex arrays.
  Complex *d, *h;
  Complex *h_out, *h_out2, *h_out3, *h_out4;
  Complex *h_transpose, *h_transpose2;
  Complex *h_new, *h_new2, *h_new3; // May have to create h_new2
  Complex *h_recv, *h_recv2, *h_recv3;
  Complex *H_2Dsend, *H_2Dsend2, *H_2Dsend3;
  Complex *A, *B, *C;
  Complex *h_1Dinv, *h_2Dinv;

  d = image.GetImageData();

  rc = MPI_Init(&argc,&argv);
	if (rc!= MPI_SUCCESS)
	{
		cout<<"Error";
		MPI_Abort(MPI_COMM_WORLD,rc);
	}

	MPI_Comm_size(MPI_COMM_WORLD,&numtasks);
	MPI_Comm_rank(MPI_COMM_WORLD,&rank);

  rowsPerCPU = ht/numtasks; // 16
  int mul = wd*rowsPerCPU;  // 16*256

  //cout<<"Numtasks: "<<numtasks<<endl;
  //cout<<"Rank: "<<rank<<"\n";

  // Declaring required complex number arrays.
  int N = image.GetWidth();

  h = new Complex[N]; // current h input array
  h_out = new Complex[N*rowsPerCPU]; // data every CPU handles.
  h_out2 = new Complex[N*rowsPerCPU]; // For 2D computation
  h_out3 = new Complex[N*rowsPerCPU];
  h_out4 = new Complex[N*rowsPerCPU];
  h_new = new Complex[N*rowsPerCPU];
  h_new2 = new Complex[N*rowsPerCPU];
  h_new3 = new Complex[N*rowsPerCPU];
  h_recv = new Complex[N*rowsPerCPU];
  h_recv2 = new Complex[N*rowsPerCPU];
  h_recv3 = new Complex[N*rowsPerCPU];
  H_2Dsend = new Complex[N*rowsPerCPU];
  H_2Dsend2 = new Complex[N*rowsPerCPU];
  H_2Dsend3 = new Complex[N*rowsPerCPU];
  h_transpose = new Complex[ht*wd]; // Transpose of 1D
  h_transpose2 = new Complex[ht*wd]; // Transpose this matrix to get final output
  A = new Complex[N*N];
  B = new Complex[N*N];
  C = new Complex[N*N];
  h_1Dinv = new Complex[ht*wd];
  h_2Dinv = new Complex[ht*wd];



  // Transforming the data every CPU handles and placing it in h_out array.
  for (int row = 0; row < rowsPerCPU; ++row)
  {
    for(int col=0; col < N; ++col)
    {
      h[col] = d[(rank*numtasks + row) *N + col];
    }
    Transform1D(h,wd,(h_out+row*N));
  //  cout<<"Rank: "<<rank<<"Finished row: "<<(rank + row*numtasks)<<endl;
  }


  // Step 1: CPU 0 writes it's own transformed data into new h_1Dout array.
  // Step 2: CPU 0 receives h_out data from other (numtasks - 1) processors

  if (rank == 0)
  {
    cout<<"Starting 1D forward transform."<<endl;
    int count = 0;
    int flag = -1;

    MPI_Status status;
    MPI_Request request;

    // Step 1:
    for(int i = 0; i < mul; i++)
    {
      A[(rank*numtasks)*N + i] = h_out[i];
    }

    // Step 2:
    while(count < 15)
    {
      if(flag != 0)   // flag is initially '0' indicating CPU0 can receivce.
      {
        MPI_Irecv(h_new, N*rowsPerCPU*sizeof(Complex),MPI_COMPLEX,MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &request);
        if (rc != MPI_SUCCESS)
        {
            cout << "Rank " << rank
                 << " recv failed, rc " << rc << endl;
            MPI_Finalize();
            //exit(1);
        }
        flag = 0;
      }
      MPI_Test(&request,&flag,&status); // Check if receive is complete
      if (flag != 0)      // After data is received flag = 1
      {
        count++;
        flag = -1; // change flag to receive again.

        for(int p = 0; p < N*rowsPerCPU; ++p)
        {
          // Transfer contents of h_new buffer to h_1Dout array depending on source.
          A[(status.MPI_SOURCE*rowsPerCPU)*N + p] = h_new[p];
        }
        //cout<<"CPU number transfered in H_1DOUT: "<<status.MPI_SOURCE<<endl;
      }
    }
  }
  // Step 3: If rank ! = 0 i.e for CPUs other than CPU0
  // Send transformed data to CPU 0
  else
  {
    rc = MPI_Send(h_out, N*rowsPerCPU*sizeof(Complex), MPI_COMPLEX, 0, 0, MPI_COMM_WORLD);
    if (rc != MPI_SUCCESS)
    {
      cout << "Rank " << rank
           << " send failed, rc " << rc << endl;
      MPI_Finalize();
    }
  }
// Step 4: CPU0 writes h_1Dout into a text file.
if (rank == 0)
{
  image.SaveImageData("MyAfter1d.txt",A,wd,ht);
  cout<<"1D forward transform done."<<endl;
  // Step 5: Transpose matrix A
  for(int row = 0; row < ht; ++row)
  {
    for(int col = 0;  col < wd; ++col)
    {
      h_transpose[row + col*ht] = A[row*wd + col];
    }
  }
  //image.SaveImageData("A_t.txt",h_transpose,wd,ht);
}

// Now, only CPU0 has the transposed matrix.
// Step 6: CPU0 calculates 1D DFT of transposed matrix and puts it in a new matrix.
// Step 7: CPU0 sends the transposed matrix to all other CPUs.
// Step 8: CPU0 receives the computed values from other CPUs
// Step 9: All other CPUs calucate the DFT and send their data to CPU0
// Step 10: CPU0 prints out 2d-DFT.

if(rank == 0)
{
  cout<<"Starting 2D forward transform."<<endl;
  Complex *h_2Din;
  h_2Din = new Complex[N*N];

  MPI_Status status;
  MPI_Request request;

  // Step 6:
  for(int row = 0; row < rowsPerCPU; ++row)
  {
    Transform1D((h_transpose + row*N),wd,(h_out2 + row*N));
  }

  for(int i = 0; i < mul; i++)
  {
    h_transpose2[(rank*numtasks) *N + i] = h_out2[i];
  }

  //image.SaveImageData("h_transpose2.txt",h_transpose2,wd,ht);

//Step 7:
  for(int cpu = 1; cpu < numtasks; ++cpu)
  {
    for(int row = 0; row < rowsPerCPU; ++row)
    {
      for(int col = 0; col < N; ++col)
      {
        h_2Din[row*N + col] = h_transpose[(cpu*numtasks + row) *N + col];
      }
    }
    rc = MPI_Send(h_2Din, N*rowsPerCPU*sizeof(Complex), MPI_COMPLEX, cpu, 0, MPI_COMM_WORLD);
    if (rc != MPI_SUCCESS)
    {
      cout << "Rank " << rank<< " send failed, rc " << rc << endl;
      MPI_Finalize();
    }
  }
  int count = 0;
  int flag = -1;

  while(count < 15)
  {
    if(flag != 0)
    {
      MPI_Irecv(h_new, N*rowsPerCPU*sizeof(Complex), MPI_COMPLEX, MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &request);
      if (rc != MPI_SUCCESS)
      {
          cout << "Rank " << rank << " recv failed, rc " << rc << endl;
          MPI_Finalize();
      }
      flag = 0;
    }
    MPI_Test(&request, &flag,  &status);
    if (flag != 0)  // After data is received correctly
    {
      count++;
      flag = -1; // change flag to receive again.
      //cout<<"Received from CPU: "<<status.MPI_SOURCE<<endl; // --> For debug
      for(int r = 0; r < mul; ++r)
      {
        h_transpose2[(status.MPI_SOURCE*rowsPerCPU)*N + r] = h_new[r];
      }

    }

  }
}
else
{
  MPI_Status status;
  rc = MPI_Recv(h_recv, N*rowsPerCPU*sizeof(Complex), MPI_COMPLEX, 0, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
  if (rc != MPI_SUCCESS)
    {
        cout << "Rank " << rank << " recv failed, rc " << rc << endl;
        MPI_Finalize();
    }

    for(int row = 0; row < rowsPerCPU; row++)
    {
      Transform1D((h_recv+row*N), wd, (H_2Dsend+row*N));
    }

    rc = MPI_Send(H_2Dsend, N*rowsPerCPU*sizeof(Complex), MPI_CHAR, 0, 0, MPI_COMM_WORLD);
    if (rc != MPI_SUCCESS)
    {
      cout << "Rank " << rank<< " send failed, rc " << rc << endl;
      MPI_Finalize();
    }

}

if(rank == 0)
{
  Complex *H_final;
  H_final = new Complex[N*N];

  for(int row = 0; row < ht; row++)
  {
    for(int col = 0; col < wd; col++)
    {
      H_final[row + col*ht] = h_transpose2[row*wd + col];
      B[row + col*ht] = h_transpose2[row*wd + col];
    }
  }
  cout<<"Forward 2D DFT done."<<endl;
  image.SaveImageData("MyAfter2d.txt",H_final,wd,ht);


}

// =================== Inverse Transform begin ================//

// CPU0 has forward DFT matrix H_final.
// Step 1: CPU0 transforms its data and puts it in H1D_inv array
// Step 2: CPU0 sends the H_final data to all other CPUs
// Step 3: CPU0 receives the transformed data from all other CPUs and then transposes it to get inverse1D.
// Step 4: All other CPUs receive the data from CPU0, transform it and sends the data back to CPU0

if (rank == 0)
{
  Complex *send2D;
  send2D = new Complex[N*N];

  MPI_Status status;
  MPI_Request request;

  cout<<"Starting 1D inverse transform.."<<endl;

  //Step 1:
  for(int row = 0; row < rowsPerCPU; ++row)
  {
    InverseTransform1D((B + row*N),wd,(h_out3 + row*N));
  }

  for(int i = 0; i < mul; i++)
  {
    h_1Dinv[(rank*numtasks) *N + i] = h_out3[i];
  }
  //image.SaveImageData("blah2.txt",h_1Dinv,wd,ht);

  //Step 2:
  for(int cpu = 1; cpu < numtasks; ++cpu)
  {
    for(int row = 0; row < rowsPerCPU; ++row)
    {
      for(int col = 0; col < N; ++col)
      {
        send2D[row*N + col] = B[(cpu*numtasks + row) *N + col];
      }
    }
      rc = MPI_Send(send2D, N*rowsPerCPU*sizeof(Complex), MPI_COMPLEX, cpu, 0, MPI_COMM_WORLD);

      if (rc != MPI_SUCCESS)
      {
        cout << "Rank " << rank<< " send failed, rc " << rc << endl;
        MPI_Finalize();
      }
  }

    int count = 0;
    int flag = -1;

    //Step 3:
    while(count < 15)
    {
      if(flag != 0)
      {
        MPI_Irecv(h_new2, N*rowsPerCPU*sizeof(Complex), MPI_COMPLEX, MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &request);
        if (rc != MPI_SUCCESS)
        {
            cout << "Rank " << rank << " recv failed, rc " << rc << endl;
            MPI_Finalize();
            //exit(1);
        }
        flag = 0;
      }
      MPI_Test(&request, &flag,  &status);
      if (flag != 0)  // After data is received correctly
      {
        count++;
        flag = -1; // change flag to receive again.
        //cout<<"Received from CPU: "<<status.MPI_SOURCE<<endl; // --> For debug
        for(int r = 0; r < mul; ++r)
        {
          h_1Dinv[(status.MPI_SOURCE*rowsPerCPU)*N + r] = h_new2[r];
        }

      }

    }

}
// Step 4:
else
{
  //cout<<"CPU: "<<rank<<"is here."<<endl;
  MPI_Status status;
  rc = MPI_Recv(h_recv2, N*rowsPerCPU*sizeof(Complex), MPI_COMPLEX, 0, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
  if (rc != MPI_SUCCESS)
    {
        cout << "Rank " << rank << " recv failed, rc " << rc << endl;
        MPI_Finalize();
        //exit(1);
    }

    for(int row = 0; row < rowsPerCPU; row++)
    {
      InverseTransform1D((h_recv2+row*N), wd, (H_2Dsend2+row*N));
    }

    rc = MPI_Send(H_2Dsend2, N*rowsPerCPU*sizeof(Complex), MPI_CHAR, 0, 0, MPI_COMM_WORLD);
    if (rc != MPI_SUCCESS)
    {
      cout << "Rank " << rank<< " send failed, rc " << rc << endl;
      MPI_Finalize();
    }
}

if(rank == 0)
{
  Complex *inv2D;
  inv2D = new Complex[N*N];

  for(int row = 0; row < ht; row++)
  {
    for(int col = 0; col < wd; col++)
    {
      inv2D[row + col*ht] = h_1Dinv[row*wd + col];
      C[row + col*ht] = h_1Dinv[row*wd + col];
    }
  }
  cout<<"Inverse 1D done."<<endl;
  //image.SaveImageData("A_inv.txt",inv2D,wd,ht);
}

// CPU0 has transformed matrix.
// Step 5: CPU0 transforms its data and puts it in H2D_inv array
// Step 6: CPU0 sends the H_final data to all other CPUs
// Step 7: CPU0 receives the transformed data from all other CPUs and then transposes it to get inverse2D.
// Step 8: All other CPUs receive the data from CPU0, transform it and sends the data back to CPU0


if (rank == 0)
{
  cout<<"Starting 2D inverse transform. "<<endl;
  Complex *send2D2;
  send2D2 = new Complex[N*N];

  MPI_Status status;
  MPI_Request request;

  //Step 5:
  for(int row = 0; row < rowsPerCPU; ++row)
  {
    InverseTransform1D((C + row*N),wd,(h_out4 + row*N));
  }

  for(int i = 0; i < mul; i++)
  {
    h_2Dinv[(rank*numtasks) *N + i] = h_out4[i];
  }

  //image.SaveImageData("B2d.txt",h_2Dinv,wd,ht);

  //Step 6:
    for(int cpu = 1; cpu < numtasks; ++cpu)
    {
      for(int row = 0; row < rowsPerCPU; ++row)
      {
        for(int col = 0; col < N; ++col)
        {
          send2D2[row*N + col] = C[(cpu*numtasks + row) *N + col];
        }
      }
      rc = MPI_Send(send2D2, N*rowsPerCPU*sizeof(Complex), MPI_COMPLEX, cpu, 0, MPI_COMM_WORLD);

      if (rc != MPI_SUCCESS)
      {
        cout << "Rank " << rank<< " send failed, rc " << rc << endl;
        MPI_Finalize();
      }
    }

    int count = 0;
    int flag = -1;

    //Step 7:
    while(count < 15)
    {
      if(flag != 0)
      {
        MPI_Irecv(h_new3, N*rowsPerCPU*sizeof(Complex), MPI_COMPLEX, MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &request);
        if (rc != MPI_SUCCESS)
        {
            cout << "Rank " << rank << " recv failed, rc " << rc << endl;
            MPI_Finalize();
        }
        flag = 0;
      }
      MPI_Test(&request, &flag,  &status);
      if (flag != 0)  // After data is received correctly
      {
        count++;
        flag = -1; // change flag to receive again.
        //cout<<"Received from CPU: "<<status.MPI_SOURCE<<endl; // --> For debug
        for(int r = 0; r < mul; ++r)
        {
          h_2Dinv[(status.MPI_SOURCE*rowsPerCPU)*N + r] = h_new3[r];
        }

      }

    }

}
// Step 8:
else
{
  //cout<<"CPU: "<<rank<<"is here"<<endl;
  MPI_Status status;
  rc = MPI_Recv(h_recv3, N*rowsPerCPU*sizeof(Complex), MPI_COMPLEX, 0, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
  if (rc != MPI_SUCCESS)
    {
        cout << "Rank " << rank << " recv failed, rc " << rc << endl;
        MPI_Finalize();
        //exit(1);
    }

    for(int row = 0; row < rowsPerCPU; row++)
    {
      InverseTransform1D((h_recv3+row*N), wd, (H_2Dsend3+row*N));
    }

    rc = MPI_Send(H_2Dsend3, N*rowsPerCPU*sizeof(Complex), MPI_CHAR, 0, 0, MPI_COMM_WORLD);
    if (rc != MPI_SUCCESS)
    {
      cout << "Rank " << rank<< " send failed, rc " << rc << endl;
      MPI_Finalize();
    }
}

if(rank == 0)
{
  Complex *inv_final;
  inv_final = new Complex[N*N];

  for(int row = 0; row < ht; row++)
  {
    for(int col = 0; col < wd; col++)
    {
      inv_final[row + col*ht] = h_2Dinv[row*wd + col];
      //C[row + col*ht] = h_1Dinv[row*wd + col];
    }
  }
  cout<<"Inverse 2D done."<<endl;
  image.SaveImageData("MyAfterInverse.txt",inv_final,wd,ht);
}

cout << "CPU: " << rank << " exiting normally.." << endl;
MPI_Finalize();

}
