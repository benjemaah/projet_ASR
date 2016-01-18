#include <complex>
#include <iostream>
#include <valarray>
#define cimg_use_tiff
#include "CImg.h"
#include <sstream>
#include <cstdlib>
#include <omp.h>
#include <sys/time.h>
#include <queue>

using namespace std;
using namespace cimg_library;

const double PI = 3.141592653589793238460;

typedef complex<double> Cmpx;

queue<Cmpx[256][256]>Q,Q1;
Cmpx c[256][256];

void conj_array(Cmpx x[256]){
	for(int i=0;i<256;i++)
		x[i]=conj(x[i]);	
}
void fft(Cmpx x[256])
{
	// DFT
	unsigned int N = 256, k = N, n;
	double thetaT = 3.14159265358979323846264338328L / N;
	Cmpx phiT = Cmpx(cos(thetaT), sin(thetaT)), T;
	while (k > 1)
	{
		n = k;
		k >>= 1;
		phiT = phiT * phiT;
		T = 1.0L;
		for (unsigned int l = 0; l < k; l++)
		{
			for (unsigned int a = l; a < N; a += n)
			{
				unsigned int b = a + k;
				Cmpx t = x[a] - x[b];
				x[a] += x[b];
				x[b] = t * T;
			}
			T *= phiT;
		}
	}
	// Decimate
	unsigned int m = (unsigned int)log2(N);
	for (unsigned int a = 0; a < N; a++)
	{
		unsigned int b = a;
		// Reverse bits
		b = (((b & 0xaaaaaaaa) >> 1) | ((b & 0x55555555) << 1));
		b = (((b & 0xcccccccc) >> 2) | ((b & 0x33333333) << 2));
		b = (((b & 0xf0f0f0f0) >> 4) | ((b & 0x0f0f0f0f) << 4));
		b = (((b & 0xff00ff00) >> 8) | ((b & 0x00ff00ff) << 8));
		b = ((b >> 16) | (b << 16)) >> (32 - m);
		if (b > a)
		{
			Cmpx t = x[a];
			x[a] = x[b];
			x[b] = t;
		}
	}
	
}

// inverse fft (in-place)
void ifft(Cmpx x[256])
{
    // conjugate the complex numbers
    conj_array(x);

    // forward fft
    fft( x );

    // conjugate the complex numbers again
    conj_array(x);

    // scale the numbers
    	for(int j=0;j<256;j++)
    		x[j] /= 256;
}

int FFT2D(int nx,int ny,int dir)
{
   int i,j;
   int m,twopm;
   double *real,*imag;
    Cmpx buf[256];

   real = (double *)malloc(nx * sizeof(double));
   imag = (double *)malloc(nx * sizeof(double));


   for (j=0;j<ny;j++) {

      for (i=0;i<nx;i++) {
         real[i] = c[i][j].real();
         imag[i] = c[i][j].imag();
	 buf[i]=c[i][j];

      }

        

      if(dir==1)fft( buf);
	else ifft( buf);

      for (i=0;i<nx;i++) {
        c[i][j]=buf[i];
      }
   }

   free(real);
   free(imag);


   real = (double *)malloc(ny * sizeof(double));
   imag = (double *)malloc(ny * sizeof(double));


   for (i=0;i<nx;i++) {
      for (j=0;j<ny;j++) {
         real[j] = c[i][j].real();
         imag[j] = c[i][j].imag();
	 buf[j]=c[i][j];
      }
      
      if(dir==1)fft( buf);
	else ifft( buf);

      for (j=0;j<ny;j++) {
         c[i][j]=buf[j];
      }
   }

   free(real);
   free(imag);

   return(true);
}



int main()
{

	rc = MPI_Init(&argc,&argv);
   if (rc != MPI_SUCCESS) {
     printf ("Error starting MPI program. Terminating.\n");
     MPI_Abort(MPI_COMM_WORLD, rc);
     }

  int rank;
  MPI_Comm_size(MPI_COMM_WORLD,&size);
  MPI_Comm_rank(MPI_COMM_WORLD,&rank);
  MPI_Get_processor_name(hostname, &len);


    		//#pragma omp parallel for schedule(dynamic)
  int marge=img_lists.size()/size;

  if(rank==0){

  	cimg_library::CImgList< unsigned char > img_lists;
    img_lists.load_tiff( "aaaa.tif" );

    cimg_library::CImg< unsigned char > top_img;
    
     struct timeval t1, t2;
  gettimeofday(&t1, NULL);

  	for(int j=0;j<=size;j++){
		for(int i=j*size;i<(j+1)*size;i++){

        top_img = img_lists[i];
        
    /***********************************creation de la matrice d'input**********************************/
    
    		//#pragma omp parallel for schedule(dynamic) collapse(2)
    for(int i=0;i<256;i++)
        for(int j=0;j<256;j++)
        	if((i>=top_img.height())||(j>=top_img.width())) c[i][j]=0;
        	else  c[i][j]=1.0*top_img.atXY(i,j);
        		
   			if(j!=0) MPI_Send(&c[0][0], 256*256 , MPI_DOUBLE_COMPLEX ,j,tag, MPI_COMM_WORLD);
   			else  Q.push(c);
    }
		}
          }
  
  if(rank!=0){
  			MPI_Recv(&c[0][0], 256*256 , MPI_DOUBLE_COMPLEX ,0,tag, MPI_COMM_WORLD, &stat); 
  			Q.push(c);
          }

    while(!Q.empty()){

    	Cmpx[256][256] c1=Q.front();
    	Q.pop();

    FFT2D(256,256,1,c1);
	
	for(int j=20;j<256;j++)
		{
           for(int i=0;i<10;i++) {c1[i][j]=0;c1[j][i]=0;}
           for(int i=250;i<256;i++) {c1[i][j]=0;c1[j][i]=0;}

		}
	

   		FFT2D(256,256,-1,c1);
   		Q1.push(c1);
    }

    if(rank!=0){
    	for(int i=1;i<=size;i++){
    		while(!Q1.empty()){
    			Cmpx c1[256][256]=Q1.front();
    			Q1.pop();
    			MPI_Send(&c1[0][0], 256*256 , MPI_DOUBLE_COMPLEX ,0,tag, MPI_COMM_WORLD);
    		}
    	}
    	
    }else{
    	
    	
    	for(int i=1;i<size;i++){
    		for(int j=0;j<marge;j++){
    			MPI_Recv(&c[0][0], 256*256 , MPI_DOUBLE_COMPLEX ,i,tag, MPI_COMM_WORLD, &stat);
    			Q.push(c);
    		}
    	}
    
	 
	while(!Q.empty()){

		c=Q.front();
		Q.pop();

    		for(int i=0;i<top_img.height();i++)
        		for(int j=0;j<top_img.width();j++) *top_img.data(i,j,0,0)=(unsigned int )c[i][j].real();
       		
     string s1 = to_string(i);
        char const *pchar = s1.c_str();
        char s[20];

        strcpy(s,pchar);
        strcat(s,".tiff");
    	}
    	gettimeofday(&t2, NULL);

  double duration = (t2.tv_sec -t1.tv_sec)+((t2.tv_usec-t1.tv_usec)/1e6);
  cout<<"duration= "<<duration<<endl;
}
	
	 
    return 0;
}



