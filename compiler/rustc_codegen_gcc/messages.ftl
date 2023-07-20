codegen_gcc_invalid_minimum_alignment =
    invalid minimum global alignment: {$err}

codegen_gcc_invalid_monomorphization_expected_simd =
    invalid monomorphization of `{$name}` intrinsic: expected SIMD {$expected_ty} type, found non-SIMD `{$found_ty}`

codegen_gcc_lto_not_supported =
    LTO is not supported. You may get a linker error.

codegen_gcc_tied_target_features = the target features {$features} must all be either enabled or disabled together
    .help = add the missing features in a `target_feature` attribute

codegen_gcc_unwinding_inline_asm =
    GCC backend does not support unwinding from inline asm
