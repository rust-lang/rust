codegen_llvm_unknown_ctarget_feature =
    unknown feature specified for `-Ctarget-feature`: `{$feature}`

codegen_llvm_unknown_feature_prefix =
    features must begin with a `+` to enable or `-` to disable it

codegen_llvm_unknown_feature =
    it is still passed through to the codegen backend

codegen_llvm_rust_feature =
    you might have meant: `{$rust_feature}`

codegen_llvm_unknown_feature_fill_request =
    consider filing a feature request

codegen_llvm_error_creating_import_library =
    Error creating import library for {$lib_name}: {$error}

codegen_llvm_instrument_coverage_requires_llvm_12 =
    rustc option `-C instrument-coverage` requires LLVM 12 or higher.

codegen_llvm_symbol_already_defined =
    symbol `{$symbol_name}` is already defined

codegen_llvm_branch_protection_requires_aarch64 =
    -Zbranch-protection is only supported on aarch64

codegen_llvm_layout_size_overflow =
    {$error}

codegen_llvm_invalid_minimum_alignment =
    invalid minimum global alignment: {$err}

codegen_llvm_linkage_const_or_mut_type =
    must have type `*const T` or `*mut T` due to `#[linkage]` attribute
