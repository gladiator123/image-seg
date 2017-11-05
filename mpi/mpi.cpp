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
#include <queue>

using namespace std;
using namespace cimg_library;

const double PI = 3.141592653589793238460;

typedef complex<double> Cmpx;

int dx[4] = { 0, 0, -1, 1 };
int dy[4] = { -1, 1, 0, 0 };

Cmpx c[256][256];
Cmpx c1[256][256];
Cmpx c2[256][256];
Cmpx c3[256][256];
Cmpx cf[256][256];
cimg_library::CImgList<unsigned char> img_lists;
cimg_library::CImg<unsigned char> top_img;
struct timeval t1, t2;

void conj_array(Cmpx x[256])
{
    for (int i = 0; i < 256; i++)
        x[i] = conj(x[i]);
}

void fft(Cmpx x[256])
{
    unsigned int N = 256, k = N, n;
    double thetaT = 3.14159265358979323846264338328L / N;
    Cmpx phiT = Cmpx(cos(thetaT), sin(thetaT)), T;
    while (k > 1) {
        n = k;
        k >>= 1;
        phiT = phiT * phiT;
        T = 1.0L;
        for (unsigned int l = 0; l < k; l++) {
            for (unsigned int a = l; a < N; a += n) {
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
    for (unsigned int a = 0; a < N; a++) {
        unsigned int b = a;
        // Reverse bits
        b = (((b & 0xaaaaaaaa) >> 1) | ((b & 0x55555555) << 1));
        b = (((b & 0xcccccccc) >> 2) | ((b & 0x33333333) << 2));
        b = (((b & 0xf0f0f0f0) >> 4) | ((b & 0x0f0f0f0f) << 4));
        b = (((b & 0xff00ff00) >> 8) | ((b & 0x00ff00ff) << 8));
        b = ((b >> 16) | (b << 16)) >> (32 - m);
        if (b > a) {
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
    fft(x);

    // conjugate the complex numbers again
    conj_array(x);

    // scale the numbers
    for (int j = 0; j < 256; j++)
        x[j] /= 256;
}

void FFT2D(int nx, int ny, int dir)
{
    Cmpx buf[256];

    for (int j = 0; j < ny; j++) {
        for (int i = 0; i < nx; i++)
            buf[i] = c[i][j];

        if (dir == 1)
            fft(buf);
        else
            ifft(buf);

        for (int i = 0; i < nx; i++) {
            c[i][j] = buf[i];
        }
    }

    for (int i = 0; i < nx; i++) {
        for (int j = 0; j < ny; j++)
            buf[j] = c[i][j];

        if (dir == 1)
            fft(buf);
        else
            ifft(buf);

        for (int j = 0; j < ny; j++) {
            c[i][j] = buf[j];
        }
    }
}

void sobel()
{
#pragma omp parallel for schedule(dynamic) collapse(2)
    for (int i = 0; i < 254; i++) {
        for (int j = 0; j < 254; j++) {
            int Gx = (2 * c[i + 2][j + 1].real() + c[i + 2][j].real() + c[i + 2][j + 2].real()) - (2 * c[i][j + 1].real() + c[i][j].real() + c[i][j + 2].real());
            int Gy = (2 * c[i + 1][j + 2].real() + c[i][j + 2].real() + c[i + 2][j + 2].real()) - (2 * c[i + 1][j].real() + c[i][j].real() + c[i + 2][j].real());
            c[i][j] = sqrt(Gx * Gx + Gy * Gy);
            if (c[i][j].real() < 70)
                c[i][j] = 0;
            else
                c[i][j] = 255;
            c1[i][j] = c[i][j]; //c1 c
        }
    }
}

void dilate()
{
#pragma omp parallel for schedule(dynamic) collapse(2)
    for (int i = 1; i < 255; i++) {
        for (int j = 1; j < 255; j++) {
            if (c[i][j].real() > 200) {
                c1[i][j] = c[i][j];
                c1[i][j] = 255;
                c1[i - 1][j] = 255;
                c1[i - 1][j + 1] = 255;
                c1[i][j - 1] = 255;
                c1[i][j + 1] = 255;
                c1[i - 1][j - 1] = 255;
                c1[i + 1][j + 1] = 255;
                c1[i - 1][j + 1] = 255;
                c1[i + 1][j - 1] = 255;
            }
        }
    }
#pragma omp parallel for schedule(dynamic) collapse(2)
    for (int i = 1; i < 255; i++)
        for (int j = 1; j < 255; j++)
            c[i][j] = c1[i][j]; //c c1
}
void erode()
{

#pragma omp parallel for schedule(dynamic) collapse(2)
    for (int i = 1; i < 255; i++) {
        for (int j = 1; j < 255; j++) {
            if (c1[i][j].real() == 0) {
                c3[i][j] = c1[i][j];
                c3[i - 1][j] = c1[i][j];
                c3[i - 1][j + 1] = c1[i][j];
                c3[i][j - 1] = c1[i][j];
                c3[i][j + 1] = c1[i][j];
            }
            else {
                c3[i][j] = c1[i][j];
            }
        }
    }
#pragma omp parallel for schedule(dynamic) collapse(2)
    for (int i = 1; i < 255; i++) {
        for (int j = 1; j < 255; j++)
            c1[i][j] = c3[i][j];
    }
}

void corners(int x, int y)
{
#pragma omp parallel for schedule(dynamic) collapse(2)
    for (int i = 0; i < 30; i++) {
        for (int j = 0; j < 10; j++) {
            c1[j][i] = 0;
            c1[j][x - i] = 0;
            c1[y - j][x - i] = 0;
            c1[y - j][i] = 0;
        }
    }
}

void filter()
{
#pragma omp parallel for schedule(dynamic) collapse(2)
    for (int j = 3; j < 256; j++) {
        for (int i = 0; i < 10; i++) {
            c[i][j] = 0;
            c[j][i] = 0;
        }
        for (int i = 250; i < 256; i++) {
            c[i][j] = 0;
            c[j][i] = 0;
        }
    }
}
void copy_mat(Cmpx c[256][256], Cmpx c1[256][256])
{
#pragma omp parallel for schedule(dynamic) collapse(2)
    for (int i = 0; i < 256; i++) {
        for (int j = 0; j < 256; j++) {
            c[i][j] = c1[i][j];
        }
    }
}

void fill()
{
    queue<pair<int, int> > Q;
    Q.push(make_pair(250, 250));
    while (!Q.empty()) {
        pair<int, int> p = Q.front();
        Q.pop();
        for (int i = 0; i < 4; i++) {
            if ((p.first + dx[i] >= 0) && (p.first + dx[i] <= 255) && (p.second + dy[i] >= 0) && (p.second + dy[i] <= 255)) {
                if (c2[p.first + dx[i]][p.second + dy[i]].real() != 255) {
                    c2[p.first + dx[i]][p.second + dy[i]] = 255;
                    Q.push(make_pair(p.first + dx[i], p.second + dy[i]));
                }
            }
        }
    }

#pragma omp parallel for schedule(dynamic) collapse(2)
    for (int i = 0; i < 255; i++) {
        for (int j = 0; j < 255; j++) {
            if ((unsigned int)c2[i][j].real() != 255) {
                c1[i][j] = 255;
            }
        }
    }
}

void read_img(int i)
{

    top_img = img_lists[i];
#pragma omp parallel for schedule(dynamic) collapse(2)
    for (int i = 0; i < 256; i++)
        for (int j = 0; j < 256; j++)
            if ((i >= top_img.height()) || (j >= top_img.width()))
                c[i][j] = 0;
            else
                c[i][j] = 1.0 * top_img.atXY(i, j);
}

void cell_extract()
{
#pragma omp parallel for schedule(dynamic) collapse(2)
    for (int i = 0; i < top_img.height(); i++) {
        for (int j = 0; j < top_img.width(); j++)
            if (c1[i][j].real() == 0)
                *top_img.data(i, j, 0, 0) = 180;
            else
                *top_img.data(i, j, 0, 0) = (unsigned int)(*top_img.data(i, j, 0, 0));
    }
}

int main()
{

    rc = MPI_Init(&argc, &argv);
    if (rc != MPI_SUCCESS) {
        printf("Error starting MPI program. Terminating.\n");
        MPI_Abort(MPI_COMM_WORLD, rc);
    }

    int rank;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Get_processor_name(hostname, &len);

    img_lists.load_tiff("aaaa.tif");
    gettimeofday(&t1, NULL);
    //#pragma omp parallel for schedule(dynamic)
    int marge = img_lists.size() / size;

    for (int j = 0; j <= size; j++) {
        if (rank == j) {
            for (int i = j * marge; i < (j + 1) * marge; i++) {

                top_img = img_lists[i];
                read_img(i);
                FFT2D(256, 256, 1);
                filter();
                FFT2D(256, 256, -1);
                copy_mat(cf, c);
                sobel();
                dilate();
                copy_mat(c2, c1);
                fill();
                corners(top_img.height(), top_img.width());
                erode();
                cell_extract();

                string s1 = to_string(i);
                char const* pchar = s1.c_str();
                char s[20];

                cimg_library::CImg<unsigned char> final_img(top_img.height(), top_img.width(), 1, 3);
#pragma omp parallel for schedule(dynamic) collapse(2)
                for (int i = 0; i < top_img.height(); i++) {
                    for (int j = 0; j < top_img.width(); j++) {

                        if (c2[i][j].real() == 0) {
                            *final_img.data(i, j, 0, 0) = 100;
                            *final_img.data(i, j, 0, 1) = 0;
                            *final_img.data(i, j, 0, 2) = 0;
                        }
                        else {
                            *final_img.data(i, j, 0, 0) = *top_img.data(i, j, 0, 0);
                            *final_img.data(i, j, 0, 1) = *top_img.data(i, j, 0, 0);
                            *final_img.data(i, j, 0, 2) = *top_img.data(i, j, 0, 0);
                        }
                    }
                }

                strcpy(s, pchar);
                strcat(s, ".tiff");
                //printf("%s\n",s);
                final_img.save_tiff(s);
            }
        }
    }

    return 0;
}

