#include <complex>
#include <iostream>
#include <valarray>
#define cimg_use_tiff
#include "CImg.h"
#include <sstream>
#include <cstdlib>
#include "dynamic_2d_array.h"
#include <omp.h>
#include <sys/time.h>

using namespace std;
using namespace cimg_library;

const double PI = 3.141592653589793238460;

typedef complex<double> Cmpx;
typedef valarray<Cmpx> CArray;

dynamic_2d_array<Cmpx > c(256,256);


void fft(CArray &x)
{
	// DFT
	unsigned int N = x.size(), k = N, n;
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
void ifft(CArray& x)
{
    // conjugate the complex numbers
    x = x.apply(std::conj);

    // forward fft
    fft( x );

    // conjugate the complex numbers again
    x = x.apply(std::conj);

    // scale the numbers
    x /= x.size();
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

        CArray data(buf, 256);

      if(dir==1)fft( data);
	else ifft( data);

      for (i=0;i<nx;i++) {
        c[i][j]=data[i];
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
        CArray data(buf, 256);
      if(dir==1)fft( data);
	else ifft( data);

      for (j=0;j<ny;j++) {
         c[i][j]=data[j];
      }
   }

   free(real);
   free(imag);

   return(true);
}



int main()
{
    cimg_library::CImgList< unsigned char > img_lists;
    img_lists.load_tiff( "aaaa.tif" );

    cimg_library::CImg< unsigned char > top_img;
    
     struct timeval t1, t2;
  gettimeofday(&t1, NULL);

 

 

    		#pragma omp parallel for schedule(dynamic)
	for(int i=0;i<2000;i++){

        top_img = img_lists[i];
        
    /***********************************creation de la matrice d'input**********************************/
    
    		//#pragma omp parallel for schedule(dynamic) collapse(2)
    for(int i=0;i<256;i++)
        for(int j=0;j<256;j++)
        //printf("%u ",top_img.atXY(i,j));
        if((i>=top_img.height())||(j>=top_img.width())) c[i][j]=0;
        		
        else  c[i][j]=1.0*top_img.atXY(i,j);
        		
    
        //cout<<c[i][j]<<" ";
    
       //cout<<endl;
  
     /**************************************************************************************************/
    FFT2D(256,256,1);
	for(int j=20;j<256;j++)
		{
           for(int i=0;i<10;i++) {c[i][j]=0;c[j][i]=0;}
           for(int i=250;i<256;i++) {c[i][j]=0;c[j][i]=0;}

		}
	
    FFT2D(256,256,-1);
    
    		//#pragma omp parallel for schedule(dynamic) collapse(2)
	for(int i=0;i<top_img.height();i++){
        for(int j=0;j<top_img.width();j++)
           *top_img.data(i,j,0,0)=(unsigned int )c[i][j].real();
       		//cout<<F[1].atXY(i,j)<<" "<<c[i][j].imag()<<" ";}
            }
     string s1 = to_string(i);
        char const *pchar = s1.c_str();
        char s[20];


        strcpy(s,pchar);
        strcat(s,".tiff");
        //printf("%s\n",s);
        //top_img.save_tiff( s );
	 


	}
	 gettimeofday(&t2, NULL);

  double duration = (t2.tv_sec -t1.tv_sec)+((t2.tv_usec-t1.tv_usec)/1e6);
  cout<<"duration= "<<duration<<endl;
    return 0;
}



