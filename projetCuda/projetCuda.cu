#include <opencv2/opencv.hpp>
#include <vector>
#include <cstring>

__global__ void ConvKernel(unsigned char* d_Result,unsigned char* d_Data,std::size_t rows,std::size_t cols) {

  auto i = blockIdx.x * blockDim.x + threadIdx.x;
  auto j = blockIdx.y * blockDim.y + threadIdx.y;
 if( i < cols && j < rows ) {
    d_Result[ j * cols + i ] = (
                         307 * d_Data[ 3 * ( j * cols + i ) ]
                         + 604 * d_Data[ 3 * ( j * cols + i ) + 1 ]
                         + 113 * d_Data[  3 * ( j * cols + i ) + 2 ]
                         ) >> 10;
  }
 }

__global__ void ConvKernelBorder( unsigned char* d_Result, unsigned char * d_Data, std::size_t cols, std::size_t rows )
        {
  auto i = blockIdx.x * blockDim.x + threadIdx.x;
  auto j = blockIdx.y * blockDim.y + threadIdx.y;

  if( i > 1 && i < cols && j > 1 && j < rows )
  {
    auto h =      d_Data[ (j-1)*cols + i - 3 ] -      d_Data[ (j-1)*cols + i + 1 ]
           + 2 *  d_Data[ (j  )*cols + i - 3 ] - 2 *  d_Data[ (j  )*cols + i + 1 ]
           +      d_Data[ (j+1)*cols + i - 3 ] -      d_Data[ (j+1)*cols + i + 1 ];

    auto v =      d_Data[ (j-1)*cols + i - 3 ] -      d_Data[ (j+1)*cols + i - 3 ]
           + 2 *  d_Data[ (j-1)*cols + i     ] - 2 *  d_Data[ (j+1)*cols + i     ]
           +    d_Data [ (j-1)*cols + i + 1 ] -      d_Data[ (j+1)*cols + i + 1 ];

    auto x = h*h + v*v;
    x = x > 65535 ? x = 65535 : x;

    d_Result[ j * cols + i ] = sqrtf( x );
  }
}
__global__ void Convkernelshared( unsigned char * d_Result, unsigned char * d_Data, std::size_t cols, std::size_t rows )
{
  auto li = threadIdx.x;
  auto lj = threadIdx.y;

  auto w = blockDim.x;
  auto h = blockDim.y;

  auto i = blockIdx.x * (blockDim.x-2) + threadIdx.x;
  auto j = blockIdx.y * (blockDim.y-2) + threadIdx.y;

  extern __shared__ unsigned char partage[];

  if( i < cols && j < rows )
  {
    partage[ lj * w + li ] =d_Data [ j * cols + i ];
  }

  __syncthreads();
  if( i < cols -1 && j < rows-1 && li > 0 && li < (w-1) && lj > 0 && lj < (h-1) )
  {
    auto h =     partage[ (lj-1)*w + li - 1 ] -     partage[ (lj-1)*w + li + 1 ]
           + 2 * partage[ (lj )*w + li - 1 ] - 2 * partage[ (lj  )*w + li + 1 ]
           +     partage[ (lj+1)*w + li - 1 ] -     partage[ (lj+1)*w + li + 1 ];

    auto v =     partage[ (lj-1)*w + li - 1 ] -     partage[ (lj+1)*w + li - 1 ]
           + 2 * partage[ (lj-1)*w + li ] - 2 * partage[(lj+1)*w + li     ]
           +     partage[ (lj-1)*w + li + 1 ] -     partage[ (lj+1)*w + li + 1 ];

    auto x = h*h + v*v;
    x = x > 65535 ? x = 65535 : x;

    d_Result[ j * cols + i ] = sqrtf( x);
  }
}
int main() {

 cv::Mat m_in = cv::imread("in.jpg", cv::IMREAD_UNCHANGED );

// auto d_cpu_Data=nullptr;

 auto rows = m_in.rows;
 auto cols = m_in.cols;
 //std::vector<unsigned char> cpu_Result(rows*cols);
unsigned char*cpu_Result=nullptr;
cudaMallocHost(&cpu_Result,rows*cols);


cv::Mat m_out( rows, cols, CV_8UC1, cpu_Result);
unsigned char * d_cpu_Data=nullptr;
cudaMallocHost(&d_cpu_Data,3*rows*cols);



 std::memcpy( d_cpu_Data, m_in.data,3* rows * cols );

 unsigned char * d_data;
 unsigned char *d_Result;
 cudaMalloc( &d_Result, rows * cols );
 cudaMalloc( &d_data, 3*rows * cols );

 cudaMemcpy( d_data, d_cpu_Data,3* rows * cols, cudaMemcpyHostToDevice );

 dim3 block( 32, 4 );
 dim3 grid0( ( cols - 1) / block.x + 1 , ( rows - 1 ) / block.y + 1 );

 dim3 grid1( ( cols - 1) / (block.x-2) + 1 , ( rows - 1 ) / (block.y-2) + 1 );


 cudaEvent_t start, stop;

 cudaEventCreate( &start );
 cudaEventCreate( &stop );

return 0;


 cudaEventRecord( start );
 ConvKernel<<< grid0, block >>>( d_Result ,d_data,rows,cols );
 // ConvKernelBorder<<<grid0,block>>>(d_Result,d_data,rows,cols);
 //Convkernelshared<<<grid1,block,block.x*block.y>>>(d_Result,d_data,rows,cols);
 cudaEventRecord(stop);

 cudaMemcpy( cpu_Result,d_Result, rows * cols, cudaMemcpyDeviceToHost );
float dure;
cudaEventElapsedTime(&dure,start,stop);

        std::cout << "time=" << dure << std::endl;

 cv::imwrite( "out2.jpg", m_out );
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  cudaFree(d_Result );
  cudaFree( d_data);


  cudaFreeHost( d_cpu_Data );
  cudaFreeHost( cpu_Result );

  return 0;
}
