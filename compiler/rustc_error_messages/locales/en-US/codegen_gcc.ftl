codegen_gcc_ranlib_failure =
    Ranlib exited with code {$exit_code}

codegen_gcc_layout_size_overflow =
    {$error}

codegen_gcc_linkage_const_or_mut_type =
    must have type `*const T` or `*mut T` due to `#[linkage]` attribute

codegen_gcc_unwinding_inline_asm =
    GCC backend does not support unwinding from inline asm

codegen_gcc_lto_not_supported =
    LTO is not supported. You may get a linker error.
