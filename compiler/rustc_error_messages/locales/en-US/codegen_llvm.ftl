codegen_llvm_unknown_ctarget_feature =
    unknown feature specified for `-Ctarget-feature`: `{$feature}`
    .note = it is still passed through to the codegen backend
    .possible_feature = you might have meant: `{$rust_feature}`
    .consider_filing_feature_request = consider filing a feature request

codegen_llvm_unknown_ctarget_feature_prefix =
    unknown feature specified for `-Ctarget-feature`: `{$feature}`
    .note = features must begin with a `+` to enable or `-` to disable it

codegen_llvm_error_creating_import_library =
    Error creating import library for {$lib_name}: {$error}

codegen_llvm_instrument_coverage_requires_llvm_12 =
    rustc option `-C instrument-coverage` requires LLVM 12 or higher.

codegen_llvm_symbol_already_defined =
    symbol `{$symbol_name}` is already defined

codegen_llvm_invalid_minimum_alignment =
    invalid minimum global alignment: {$err}

codegen_llvm_linkage_const_or_mut_type =
    must have type `*const T` or `*mut T` due to `#[linkage]` attribute

codegen_llvm_sanitizer_memtag_requires_mte =
    `-Zsanitizer=memtag` requires `-Ctarget-feature=+mte`

codegen_llvm_error_writing_def_file =
    Error writing .DEF file: {$error}

codegen_llvm_error_calling_dlltool =
    Error calling dlltool: {$error}

codegen_llvm_dlltool_fail_import_library =
    Dlltool could not create import library: {$stdout}\n{$stderr}

codegen_llvm_target_feature_disable_or_enable =
    the target features {$features} must all be either enabled or disabled together

codegen_llvm_missing_features =
    add the missing features in a `target_feature` attribute

codegen_llvm_dynamic_linking_with_lto =
    cannot prefer dynamic linking when performing LTO
    .note = only 'staticlib', 'bin', and 'cdylib' outputs are supported with LTO

codegen_llvm_fail_parsing_target_machine_config_to_target_machine =
    failed to parse target machine config to target machine: {$error}
