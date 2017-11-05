#include <complex>
#include <iostream>
#include <valarray>
#include <sstream>
#include <cstdlib>
#include <cufft.h>
#include <omp.h>
#include <sys/time.h>
#include <queue>
#include "tiffio.h"
#include <math.h>
#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <cuda.h>

using namespace std;

typedef complex<double> Cmpx;
int SIGNAL_SIZE = 256 * 256;
float c[256][256];

int dx[4] = { 0, 0, -1, 1 };
int dy[4] = { -1, 1, 0, 0 };

__global__ void border(cufftComplex* c1, int x, int y)
{

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (((i < 256) && (i >= x) && (j < 256) && (j >= 0)) || ((i < 256) && (i >= 0) && (j < 256) && (j >= y))) {
        c1[i * 256 + j].x = 0;
    }
}

__global__ void dilate(cufftComplex* c1, cufftComplex* c2)
{

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if ((i < 255) && (j < 255) && (i >= 1) && (j >= 1)) {
        if (c1[i * 256 + j].x > 200) {
            c2[i * 256 + j].x = c1[i * 256 + j].x;
            c2[i * 256 + j].x = 255;
            c2[(i - 1) * 256 + j].x = 255;
            c2[(i - 1) * 256 + j + 1].x = 255;
            c2[i * 256 + j - 1].x = 255;
            c2[i * 256 + j + 1].x = 255;
            c2[(i - 1) * 256 + j - 1].x = 255;
            c2[(i + 1) * 256 + j + 1].x = 255;
            c2[(i - 1) * 256 + j + 1].x = 255;
            c2[(i + 1) * 256 + j - 1].x = 255;
        }
    }
}

__global__ void sobel(cufftComplex* c1, cufftComplex* c2)
{

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if ((i < 254) && (j < 254) && (i >= 0) && (j >= 0)) {

        float Gx = (2 * c1[(i + 2) * 256 + j + 1].x + c1[(i + 2) * 256 + j].x + c1[(i + 2) * 256 + j + 2].x) - (2 * c1[i * 256 + j + 1].x + c1[i * 256 + j].x + c1[i * 256 + j + 2].x);
        float Gy = (2 * c1[(i + 1) * 256 + j + 2].x + c1[i * 256 + j + 2].x + c1[(i + 2) * 256 + j + 2].x) - (2 * c1[(i + 1) * 256 + j].x + c1[i * 256 + j].x + c1[(i + 2) * 256 + j].x);
        c2[i * 256 + j].x = sqrt(Gx * Gx + Gy * Gy);
        if (c2[i * 256 + j].x < 70)
            c2[i * 256 + j].x = 0;
        else
            c2[i * 256 + j].x = 255;
    }
}

__global__ void divide(cufftComplex* c1, int SIGNAL_SIZE)
{

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if ((i < 256) && (j < 256)) {

        c1[i * 256 + j].x = c1[i * 256 + j].x / (float)SIGNAL_SIZE;
        c1[i * 256 + j].y = c1[i * 256 + j].y / (float)SIGNAL_SIZE;
    }
}

__global__ void filter(cufftComplex* d_signal)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if ((j < 256) && (j >= 3)) {
        if (((i < 10) && (i >= 0)) || ((i < 256) && (i >= 250))) {
            d_signal[i * 256 + j].x = 0;
            d_signal[j * 256 + i].x = 0;
            d_signal[i * 256 + j].y = 0;
            d_signal[j * 256 + i].y = 0;
        }
    }
}

int main()
{
    TIFF* tif = TIFFOpen("aaaa.tif", "r");

    struct timeval t1, t2;
    gettimeofday(&t1, NULL);
    int comp = 0;

    int mem_size = sizeof(cufftComplex) * SIGNAL_SIZE;
    // Allocate device memory for signal
    cufftComplex* d_signal;
    cufftComplex* d_signal1;
    cufftComplex* d_signal2;
    checkCudaErrors(cudaMalloc((void**)&d_signal, mem_size));
    checkCudaErrors(cudaMalloc((void**)&d_signal1, mem_size));
    checkCudaErrors(cudaMalloc((void**)&d_signal2, mem_size));
    cufftHandle plan, invplan;
    cufftPlan2d(&plan, 256, 256, CUFFT_C2C);
    cufftPlan2d(&invplan, 256, 256, CUFFT_C2C);
    cufftComplex* h_inverse_signal = (cufftComplex*)malloc(sizeof(cufftComplex) * SIGNAL_SIZE);
    cufftComplex* h_inverse_signal2 = (cufftComplex*)malloc(sizeof(cufftComplex) * SIGNAL_SIZE);

    do {
        uint32 imagelength;
        tsize_t scanline;
        tdata_t buf;
        uint32 row;
        uint32 col;
        uint8* data;

        TIFFGetField(tif, TIFFTAG_IMAGELENGTH, &imagelength);
        scanline = TIFFScanlineSize(tif);
        buf = _TIFFmalloc(scanline);
        for (row = 0; row < imagelength; row++) {
            TIFFReadScanline(tif, buf, row);
            for (col = 0; col < scanline; col++) {
                data = (uint8*)buf;

                c[row][col] = data[col];
            }
        }
        // #pragma omp parallel for schedule(dynamic) collapse(2)
        for (int i = 0; i < 256; i++)
            for (int j = 0; j < 256; j++)
                if ((i >= imagelength) || (j >= scanline))
                    c[i][j] = 0;

        cufftComplex* h_signal = (cufftComplex*)malloc(sizeof(cufftComplex) * SIGNAL_SIZE);

        // #pragma omp parallel for schedule(dynamic) collapse(2)
        for (unsigned int k1 = 0; k1 < 256; ++k1) {
            for (int k2 = 0; k2 < 256; k2++) {
                h_signal[k1 * 256 + k2].x = c[k1][k2];
                h_signal[k1 * 256 + k2].y = 0;
            }
        }

        // Copy host memory to device
        checkCudaErrors(cudaMemcpy(d_signal, h_signal, mem_size, cudaMemcpyHostToDevice));

        // CUFFT plan

        // Transform signal
        // printf("Transforming signal cufftExecC2C\n");
        checkCudaErrors(cufftExecC2C(plan, (cufftComplex*)d_signal, (cufftComplex*)d_signal, CUFFT_FORWARD));

        dim3 nbthreadsbyblock(32, 32);
        dim3 nbblocksbygrid(128, 128);
        filter<<<nbblocksbygrid, nbthreadsbyblock> > >(d_signal);

        // Transform signal back
        // printf("Transforming signal back cufftExecC2C\n");
        checkCudaErrors(cufftExecC2C(invplan, (cufftComplex*)d_signal, (cufftComplex*)d_signal, CUFFT_INVERSE));

        divide<<<nbblocksbygrid, nbthreadsbyblock> > >(d_signal, SIGNAL_SIZE);
        sobel<<<nbblocksbygrid, nbthreadsbyblock> > >(d_signal, d_signal1);
        dilate<<<nbblocksbygrid, nbthreadsbyblock> > >(d_signal1, d_signal2);
        border<<<nbblocksbygrid, nbthreadsbyblock> > >(d_signal2, imagelength, scanline);

        // Copy device memory to host

        checkCudaErrors(cudaMemcpy(h_inverse_signal, d_signal2, mem_size, cudaMemcpyDeviceToHost));
        checkCudaErrors(cudaMemcpy(h_inverse_signal2, d_signal2, mem_size, cudaMemcpyDeviceToHost));

        queue<pair<int, int> > Q;
        Q.push(make_pair(250, 250));
        while (!Q.empty()) {
            pair<int, int> p = Q.front();
            Q.pop();
            for (int i = 0; i < 4; i++) {
                if ((p.first + dx[i] >= 0) && (p.first + dx[i] <= 255) && (p.second + dy[i] >= 0) && (p.second + dy[i] <= 255)) {
                    if (h_inverse_signal2[(p.first + dx[i]) * 256 + p.second + dy[i]].x < 255) {
                        h_inverse_signal2[(p.first + dx[i]) * 256 + p.second + dy[i]].x = 255;
                        Q.push(make_pair(p.first + dx[i], p.second + dy[i]));
                    }
                }
            }
        }
        //#pragma omp parallel for schedule(dynamic) collapse(2)
        for (int i = 0; i < 255; i++) {
            for (int j = 0; j < 255; j++) {
                if ((unsigned int)h_inverse_signal2[i * 256 + j].x < 255) {
                    h_inverse_signal[i * 256 + j].x = 255;
                }
                if (h_inverse_signal[i * 256 + j].x != 255) {
                    c[i][j] = 180;
                }
            }
        }

        char* m_data_cropped = (char*)malloc(sizeof(char) * scanline * imagelength * 3);

        //#pragma omp parallel for schedule(dynamic) collapse(2)
        for (int i = 0; i < imagelength; i++) {
            for (int j = 0; j < scanline; j++) {
                if (h_inverse_signal2[i * 256 + j].x == 0) {
                    m_data_cropped[3 * (i * scanline + j)] = 100;
                    m_data_cropped[3 * (i * scanline + j) + 1] = 0;
                    m_data_cropped[3 * (i * scanline + j) + 2] = 0;
                }
                else {
                    m_data_cropped[3 * (i * scanline + j)] = c[i][j];
                    m_data_cropped[3 * (i * scanline + j) + 1] = c[i][j];
                    m_data_cropped[3 * (i * scanline + j) + 2] = c[i][j];
                }
            }
        }

        string s1 = to_string(comp);
        char const* pchar = s1.c_str();
        char s[20];
        strcpy(s, pchar);
        strcat(s, ".tif");

        TIFF* tif1 = TIFFOpen(s, "w");
        TIFFSetField(tif1, TIFFTAG_IMAGEWIDTH, scanline);
        TIFFSetField(tif1, TIFFTAG_IMAGELENGTH, imagelength);
        TIFFSetField(tif1, TIFFTAG_SAMPLESPERPIXEL, 3);
        TIFFSetField(tif1, TIFFTAG_BITSPERSAMPLE, 8);
        TIFFSetField(tif1, TIFFTAG_PLANARCONFIG, PLANARCONFIG_CONTIG);

        if ((TIFFWriteEncodedStrip(tif1, 0, &m_data_cropped[0], scanline * imagelength * 3)) == -1) {
            //std::cerr << "Unable to write tif file: " << "image.tif" << std::endl;
        }
        else {
            //std::cout << "Image is saved! size is : " << image_s << std::endl;
        }

        TIFFClose(tif1);
        comp++;
        TIFFWriteDirectory(tif);
    } while (comp < 2000);

    cudaFree(d_signal);
    cudaFree(d_signal1);
    cudaFree(d_signal2);
    gettimeofday(&t2, NULL);

    double duration = (t2.tv_sec - t1.tv_sec) + ((t2.tv_usec - t1.tv_usec) / 1e6);
    cout << "duration= " << duration << endl;

    return 0;
}

