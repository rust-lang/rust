codegen_gcc_unknown_ctarget_feature_prefix =
    unknown feature specified for `-Ctarget-feature`: `{$feature}`
    .note = features must begin with a `+` to enable or `-` to disable it

codegen_gcc_invalid_minimum_alignment =
    invalid minimum global alignment: {$err}

codegen_gcc_lto_not_supported =
    LTO is not supported. You may get a linker error.

codegen_gcc_unwinding_inline_asm =
    GCC backend does not support unwinding from inline asm

codegen_gcc_copy_bitcode = failed to copy bitcode to object file: {$err}

codegen_gcc_dynamic_linking_with_lto =
    cannot prefer dynamic linking when performing LTO
    .note = only 'staticlib', 'bin', and 'cdylib' outputs are supported with LTO

codegen_gcc_lto_disallowed = lto can only be run for executables, cdylibs and static library outputs

codegen_gcc_lto_dylib = lto cannot be used for `dylib` crate type without `-Zdylib-lto`

codegen_gcc_lto_bitcode_from_rlib = failed to get bitcode from object file for LTO ({$gcc_err})

codegen_gcc_unknown_ctarget_feature =
    unknown feature specified for `-Ctarget-feature`: `{$feature}`
    .note = it is still passed through to the codegen backend
    .possible_feature = you might have meant: `{$rust_feature}`
    .consider_filing_feature_request = consider filing a feature request

codegen_gcc_missing_features =
    add the missing features in a `target_feature` attribute

codegen_gcc_target_feature_disable_or_enable =
    the target features {$features} must all be either enabled or disabled together
