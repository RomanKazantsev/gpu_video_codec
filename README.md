## Device

Device 0: GeForce GTX 1060 with Max-Q Design

Compute Capability: 6.1

Number of multiprocessors: 10

Total amount of constant memory: 64.00 KB

Total amount of global memory: 6291456.00 KB

Total amount of shared memory per block: 48.00 KB

Warp size: 32

Maximum number of threads per block: 1024

## Performance Results

| Version             | Single-thread impl. | OpenMP impl. | GPU impl. without copy | GPU impl. with copy | GPU Copy operation |
|---------------------|---------------------|--------------|------------------------|---------------------|--------------------|
| baseline            | 0.0039335s          | 0.0019126s   | 0.0001628s             | 0.0013599s          | 0.0011971s         |
| +pinned host memory | xxxxxxxxxx          | xxxxxxxxxx   | 0.0001362s             | 0.0008509s          | 0.0007147s         |


