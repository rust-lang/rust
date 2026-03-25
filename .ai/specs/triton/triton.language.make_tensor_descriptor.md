Triton

Getting Started

-   Installation
-   Tutorials

Python API

-   triton
-   triton.language
    -   Programming Model
    -   Creation Ops
    -   Shape Manipulation Ops
    -   Linear Algebra Ops
    -   Memory/Pointer Ops
        -   triton.language.load
        -   triton.language.store
        -   triton.language.make_tensor_descriptor
            -   `make_tensor_descriptor()`
        -   triton.language.load_tensor_descriptor
        -   triton.language.store_tensor_descriptor
        -   triton.language.make_block_ptr
        -   triton.language.advance
    -   Indexing Ops
    -   Math Ops
    -   Reduction Ops
    -   Scan/Sort Ops
    -   Atomic Ops
    -   Random Number Generation
    -   Iterators
    -   Inline Assembly
    -   Compiler Hint Ops
    -   Debug Ops
-   triton.testing
-   Triton Semantics

Triton MLIR Dialects

-   Triton MLIR Dialects and Ops

Programming Guide

-   Introduction
-   Related Work
-   Debugging Triton

Triton

-   
-   triton.language
-   triton.language.make_tensor_descriptor
-   View page source

------------------------------------------------------------------------

# triton.language.make_tensor_descriptor¶

triton.language.make_tensor_descriptor(*base: tensor*, *shape: List\[tensor\]*, *strides: List\[tensor\]*, *block_shape: List\[constexpr\]*, *padding_option=\'zero\'*, *\_semantic=None*) → tensor_descriptor¶

:   Make a tensor descriptor object

    Parameters:

    :   -   **base** -- the base pointer of the tensor, must be 16-byte aligned

        -   **shape** -- A list of non-negative integers representing the tensor shape

        -   **strides** -- A list of tensor strides. Leading dimensions must be multiples of 16-byte strides and the last dimension must be contiguous.

        -   **block_shape** -- The shape of block to be loaded/stored from global memory

    Notes

    On NVIDIA GPUs with TMA support, this will result in a TMA descriptor object and loads and stores from the descriptor will be backed by the TMA hardware.

    Currently only 2-5 dimensional tensors are supported.

    Example

        @triton.jit
        def inplace_abs(in_out_ptr, M, N, M_BLOCK: tl.constexpr, N_BLOCK: tl.constexpr):
            desc = tl.make_tensor_descriptor(
                in_out_ptr,
                shape=[M, N],
                strides=[N, 1],
                block_shape=[M_BLOCK, N_BLOCK],
            )

            moffset = tl.program_id(0) * M_BLOCK
            noffset = tl.program_id(1) * N_BLOCK

            value = desc.load([moffset, noffset])
            desc.store([moffset, noffset], tl.abs(value))

        # TMA descriptors require a global memory allocation
        def alloc_fn(size: int, alignment: int, stream: Optional[int]):
            return torch.empty(size, device="cuda", dtype=torch.int8)

        triton.set_allocator(alloc_fn)

        M, N = 256, 256
        x = torch.randn(M, N, device="cuda")
        M_BLOCK, N_BLOCK = 32, 32
        grid = (M / M_BLOCK, N / N_BLOCK)
        inplace_abs[grid](x, M, N, M_BLOCK, N_BLOCK)

Previous Next

------------------------------------------------------------------------

© Copyright 2020, Philippe Tillet.

Built with Sphinx using a theme provided by Read the Docs.
