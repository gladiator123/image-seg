CC=gcc

CUDA_HOME=/usr/local/cuda-7.0
CUDASDK=/usr/local/cuda-7.0/samples
CUDANVIDIA=/usr/lib64/nvidia
NVCC=$(CUDA_HOME)/bin/nvcc
NNN=/usr/local/cuda-7.0/targets/x86_64-linux/lib/
TIFI=/mci/ei1417/hmissa_m/tiff/include
TIFL=/mci/ei1417/hmissa_m/tiff/lib

NVCFLAGS= -arch=sm_21 -I $(CUDASDK)/common/inc -I $(CUDA_HOME)/include -I$(TIFI)

CFLAGS=-arch=sm_21 -L$(CUDA_HOME)/lib64 -L$(CUDANVIDIA) -L$(NNN) -L$(TIFL)

exp: exp.o
	$(NVCC) -std=c++11 $(CFLAGS) -o $@ $^ -lcuda -lrt -lcufft -ltiff -Xcompiler -fopenmp

%.o: %.cu
	$(NVCC) -std=c++11 $(NVCFLAGS) -c -o $@ $<   

clean:
	rm -f *.o *.linkinfo exp
