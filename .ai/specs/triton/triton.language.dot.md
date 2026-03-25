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
            -   `dot()`
        -   triton.language.dot_scaled
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
-   triton.language.dot
-   View page source

------------------------------------------------------------------------

# triton.language.dot¶

triton.language.dot(*input*, *other*, *acc=None*, *input_precision=None*, *allow_tf32=None*, *max_num_imprecise_acc=None*, *out_dtype=triton.language.float32*, *\_semantic=None*)¶

:   Returns the matrix product of two blocks.

    The two blocks must both be two-dimensional or three-dimensional and have compatible inner dimensions. For three-dimensional blocks, tl.dot performs the batched matrix product, where the first dimension of each block represents the batch dimension.

    Warning

    When using TF32 precision, the float32 inputs may be truncated to TF32 format (19-bit floating point) without rounding which may bias the result. For best results, you must round to TF32 explicitly, or load the data using TensorDescriptor with round_f32_to_tf32=True.

    Parameters:

    :   -   **input** (2D or 3D tensor of scalar-type in {`int8`, `float8_e5m2`, `float16`, `bfloat16`, `float32`}) -- The first tensor to be multiplied.

        -   **other** (2D or 3D tensor of scalar-type in {`int8`, `float8_e5m2`, `float16`, `bfloat16`, `float32`}) -- The second tensor to be multiplied.

        -   **acc** (2D or 3D tensor of scalar-type in {`float16`, `float32`, `int32`}) -- The accumulator tensor. If not None, the result is added to this tensor.

        -   **input_precision** (string. Available options for nvidia: `"tf32"`, `"tf32x3"`, `"ieee"`. Default: `"tf32"`. Available options for amd: `"ieee"`, (CDNA3 only) `"tf32"`.) -- How to exercise the Tensor Cores for f32 x f32. If the device does not have Tensor Cores or the inputs are not of dtype f32, this option is ignored. For devices that do have tensor cores, the default precision is tf32.

        -   **allow_tf32** -- *Deprecated.* If true, input_precision is set to "tf32". Only one of `input_precision` and `allow_tf32` can be specified (i.e. at least one must be `None`).

Previous Next

------------------------------------------------------------------------

© Copyright 2020, Philippe Tillet.

Built with Sphinx using a theme provided by Read the Docs.
