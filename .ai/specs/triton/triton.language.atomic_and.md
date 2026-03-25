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
    -   Indexing Ops
    -   Math Ops
    -   Reduction Ops
    -   Scan/Sort Ops
    -   Atomic Ops
        -   triton.language.atomic_add
        -   triton.language.atomic_and
            -   `atomic_and()`
        -   triton.language.atomic_cas
        -   triton.language.atomic_max
        -   triton.language.atomic_min
        -   triton.language.atomic_or
        -   triton.language.atomic_xchg
        -   triton.language.atomic_xor
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
-   triton.language.atomic_and
-   View page source

------------------------------------------------------------------------

# triton.language.atomic_and¶

triton.language.atomic_and(*pointer*, *val*, *mask=None*, *sem=None*, *scope=None*, *\_semantic=None*)¶

:   Performs an atomic logical and at the memory location specified by `pointer`.

    Return the data stored at `pointer` before the atomic operation.

    Parameters:

    :   -   **pointer** (*Block* *of* *dtype=triton.PointerDType*) -- The memory locations to operate on

        -   **val** (*Block* *of* *dtype=pointer.dtype.element_ty*) -- The values with which to perform the atomic operation

        -   **sem** (*str,* *optional*) -- Specifies the memory semantics for the operation. Acceptable values are "acquire", "release", "acq_rel" (stands for "ACQUIRE_RELEASE"), and "relaxed". If not provided, the function defaults to using "acq_rel" semantics.

        -   **scope** (*str,* *optional*) -- Defines the scope of threads that observe the synchronizing effect of the atomic operation. Acceptable values are "gpu" (default), "cta" (cooperative thread array, thread block), or "sys" (stands for "SYSTEM"). The default value is "gpu".

    This function can also be called as a member function on `tensor`, as `x.atomic_and(...)` instead of `atomic_and(x,`` ``...)`.

Previous Next

------------------------------------------------------------------------

© Copyright 2020, Philippe Tillet.

Built with Sphinx using a theme provided by Read the Docs.
