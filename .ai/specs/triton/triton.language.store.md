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
            -   `store()`
        -   triton.language.make_tensor_descriptor
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
-   triton.language.store
-   View page source

------------------------------------------------------------------------

# triton.language.store¶

triton.language.store(*pointer*, *value*, *mask=None*, *boundary_check=()*, *cache_modifier=\'\'*, *eviction_policy=\'\'*, *\_semantic=None*)¶

:   Store a tensor of data into memory locations defined by pointer.

    > 1.  If pointer is a single element pointer, a scalar is stored. In this case:
    >
    >     -   mask must also be scalar, and
    >
    >     -   boundary_check and padding_option must be empty.
    >
    > 2.  If pointer is an N-dimensional tensor of pointers, an N-dimensional block is stored. In this case:
    >
    >     -   mask is implicitly broadcast to pointer.shape, and
    >
    >     -   boundary_check must be empty.
    >
    > 3.  If pointer is a block pointer defined by make_block_ptr, a block of data is stored. In this case:
    >
    >     -   mask must be None, and
    >
    >     -   boundary_check can be specified to control the behavior of out-of-bound access.

    value is implicitly broadcast to pointer.shape and typecast to pointer.dtype.element_ty.

    Parameters:

    :   -   **pointer** (triton.PointerType, or block of dtype=triton.PointerType) -- The memory location where the elements of value are stored

        -   **value** (*Block*) -- The tensor of elements to be stored

        -   **mask** (*Block* *of* *triton.int1,* *optional*) -- If mask\[idx\] is false, do not store value\[idx\] at pointer\[idx\]

        -   **boundary_check** (*tuple* *of* *ints,* *optional*) -- tuple of integers, indicating the dimensions which should do the boundary check

        -   **cache_modifier** (str, optional, should be one of {"", ".wb", ".cg", ".cs", ".wt"}, where ".wb" stands for cache write-back all coherent levels, ".cg" stands for cache global, ".cs" stands for cache streaming, ".wt" stands for cache write-through, see cache operator for more details.) -- changes cache option in NVIDIA PTX

        -   **eviction_policy** (*str,* *optional,* *should be one* *of* *{\"\",* *\"evict_first\",* *\"evict_last\"}*) -- changes eviction policy in NVIDIA PTX

    This function can also be called as a member function on `tensor`, as `x.store(...)` instead of `store(x,`` ``...)`.

Previous Next

------------------------------------------------------------------------

© Copyright 2020, Philippe Tillet.

Built with Sphinx using a theme provided by Read the Docs.
