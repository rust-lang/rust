Triton

Getting Started

-   Installation
-   Tutorials

Python API

-   triton
    -   triton.jit
    -   triton.autotune
    -   triton.heuristics
    -   triton.Config
        -   `Config`
            -   `Config.__init__()`
-   triton.language
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
-   triton
-   triton.Config
-   View page source

------------------------------------------------------------------------

# triton.Config¶

*class *triton.Config(*self*, *kwargs*, *num_warps=4*, *num_stages=3*, *num_ctas=1*, *maxnreg=None*, *pre_hook=None*, *ir_override=None*)¶

:   An object that represents a possible kernel configuration for the auto-tuner to try.

    Variables:

    :   -   **kwargs** -- a dictionary of meta-parameters to pass to the kernel as keyword arguments.

        -   **num_warps** -- the number of warps to use for the kernel when compiled for GPUs. For example, if num_warps=8, then each kernel instance will be automatically parallelized to cooperatively execute using 8 \* 32 = 256 threads.

        -   **num_stages** -- the number of stages that the compiler should use when software-pipelining loops. Mostly useful for matrix multiplication workloads on SM80+ GPUs.

        -   **num_ctas** -- number of blocks in a block cluster. SM90+ only.

        -   **maxnreg** -- maximum number of registers one thread can use. Corresponds to ptx .maxnreg directive. Not supported on all platforms.

        -   **pre_hook** -- a function that will be called before the kernel is called. Parameters of this function are args.

        -   **ir_override** -- filename of a user-defined IR (\*.{ttgir\|llir\|ptx\|amdgcn}).

    \_\_init\_\_(*self*, *kwargs*, *num_warps=4*, *num_stages=3*, *num_ctas=1*, *maxnreg=None*, *pre_hook=None*, *ir_override=None*)¶

    :   

    Methods

      ----------------------------------------------- --
      `__init__`(self, kwargs\[, num_warps, \...\])   
      `all_kwargs`(self)                              
      ----------------------------------------------- --

Previous Next

------------------------------------------------------------------------

© Copyright 2020, Philippe Tillet.

Built with Sphinx using a theme provided by Read the Docs.
