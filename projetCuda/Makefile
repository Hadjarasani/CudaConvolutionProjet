CXX=g++
CXXFLAGS=-O3 -march=native
LDLIBS1=-lm -lIL
LDLIBS2=`pkg-config --libs --cflags opencv`


all: projetCuda-cu



projetCuda-cu: projetCuda.cu
        nvcc -o $@ $< $(LDLIBS2)

.PHONY: clean

clean:
        rm projetCuda projetCuda-cu
