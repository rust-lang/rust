codegen_gcc_unwinding_inline_asm =
    GCC backend does not support unwinding from inline asm

codegen_gcc_copy_bitcode = failed to copy bitcode to object file: {$err}

codegen_gcc_dynamic_linking_with_lto =
    cannot prefer dynamic linking when performing LTO
    .note = only 'staticlib', 'bin', and 'cdylib' outputs are supported with LTO

codegen_gcc_lto_disallowed = lto can only be run for executables, cdylibs and static library outputs

codegen_gcc_lto_dylib = lto cannot be used for `dylib` crate type without `-Zdylib-lto`

codegen_gcc_lto_bitcode_from_rlib = failed to get bitcode from object file for LTO ({$gcc_err})
