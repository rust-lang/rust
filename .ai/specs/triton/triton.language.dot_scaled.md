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
        -   triton.language.dot
        -   triton.language.dot_scaled
            -   `dot_scaled()`
    -   Memory/Pointer Ops
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
-   triton.language.dot_scaled
-   View page source

------------------------------------------------------------------------

# triton.language.dot_scaled¶

triton.language.dot_scaled(*lhs*, *lhs_scale*, *lhs_format*, *rhs*, *rhs_scale*, *rhs_format*, *acc=None*, *fast_math=False*, *lhs_k_pack=True*, *rhs_k_pack=True*, *out_dtype=triton.language.float32*, *\_semantic=None*)¶

:   Returns the matrix product of two blocks in microscaling format.

    lhs and rhs use microscaling formats described here: https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf

    Software emulation enables targeting hardware architectures without native microscaling operation support. Right now for such case, microscaled lhs/rhs are upcasted to `bf16` element type beforehand for dot computation, with one exception: for AMD CDNA3 specifically, if one of the inputs is of `fp16` element type, the other input is also upcasted to `fp16` element type instead. This behavior is experimental and may be subject to change in the future.

    Parameters:

    :   -   **lhs** (*2D tensor representing fp4,* *fp8* *or* *bf16 elements. Fp4 elements are packed into uint8 inputs with the first element in lower bits. Fp8 are stored as uint8* *or* *the corresponding fp8 type.*) -- The first tensor to be multiplied.

        -   **lhs_scale** (*e8m0 type represented as an uint8 tensor, or* *None.*) -- Scale factor for lhs tensor. Shape should be \[M, K//group_size\] when lhs is \[M, K\], where group_size is 32 if scales type are e8m0.

        -   **lhs_format** (*str*) -- format of the lhs tensor. Available formats: {`e2m1`, `e4m3`, `e5m2`, `bf16`, `fp16`}.

        -   **rhs** (*2D tensor representing fp4,* *fp8* *or* *bf16 elements. Fp4 elements are packed into uint8 inputs with the first element in lower bits. Fp8 are stored as uint8* *or* *the corresponding fp8 type.*) -- The second tensor to be multiplied.

        -   **rhs_scale** (*e8m0 type represented as an uint8 tensor, or* *None.*) -- Scale factor for rhs tensor. Shape should be \[N, K//group_size\] where rhs is \[K, N\]. Important: Do NOT transpose rhs_scale

        -   **rhs_format** (*str*) -- format of the rhs tensor. Available formats: {`e2m1`, `e4m3`, `e5m2`, `bf16`, `fp16`}.

        -   **acc** -- The accumulator tensor. If not None, the result is added to this tensor.

        -   **lhs_k_pack** (*bool,* *optional*) -- If false, the lhs tensor is packed into uint8 along M dimension.

        -   **rhs_k_pack** (*bool,* *optional*) -- If false, the rhs tensor is packed into uint8 along N dimension.

Previous Next

------------------------------------------------------------------------

© Copyright 2020, Philippe Tillet.

Built with Sphinx using a theme provided by Read the Docs.
