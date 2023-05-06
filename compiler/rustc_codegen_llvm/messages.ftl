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

codegen_llvm_symbol_already_defined =
    symbol `{$symbol_name}` is already defined

codegen_llvm_invalid_minimum_alignment =
    invalid minimum global alignment: {$err}

codegen_llvm_sanitizer_memtag_requires_mte =
    `-Zsanitizer=memtag` requires `-Ctarget-feature=+mte`

codegen_llvm_error_writing_def_file =
    Error writing .DEF file: {$error}

codegen_llvm_error_calling_dlltool =
    Error calling dlltool '{$dlltool_path}': {$error}

codegen_llvm_dlltool_fail_import_library =
    Dlltool could not create import library: {$stdout}
    {$stderr}

codegen_llvm_target_feature_disable_or_enable =
    the target features {$features} must all be either enabled or disabled together

codegen_llvm_missing_features =
    add the missing features in a `target_feature` attribute

codegen_llvm_dynamic_linking_with_lto =
    cannot prefer dynamic linking when performing LTO
    .note = only 'staticlib', 'bin', and 'cdylib' outputs are supported with LTO

codegen_llvm_parse_target_machine_config =
    failed to parse target machine config to target machine: {$error}

codegen_llvm_lto_disallowed = lto can only be run for executables, cdylibs and static library outputs

codegen_llvm_lto_dylib = lto cannot be used for `dylib` crate type without `-Zdylib-lto`

codegen_llvm_lto_bitcode_from_rlib = failed to get bitcode from object file for LTO ({$llvm_err})

codegen_llvm_write_output = could not write output to {$path}
codegen_llvm_write_output_with_llvm_err = could not write output to {$path}: {$llvm_err}

codegen_llvm_target_machine = could not create LLVM TargetMachine for triple: {$triple}
codegen_llvm_target_machine_with_llvm_err = could not create LLVM TargetMachine for triple: {$triple}: {$llvm_err}

codegen_llvm_run_passes = failed to run LLVM passes
codegen_llvm_run_passes_with_llvm_err = failed to run LLVM passes: {$llvm_err}

codegen_llvm_serialize_module = failed to serialize module {$name}
codegen_llvm_serialize_module_with_llvm_err = failed to serialize module {$name}: {$llvm_err}

codegen_llvm_write_ir = failed to write LLVM IR to {$path}
codegen_llvm_write_ir_with_llvm_err = failed to write LLVM IR to {$path}: {$llvm_err}

codegen_llvm_prepare_thin_lto_context = failed to prepare thin LTO context
codegen_llvm_prepare_thin_lto_context_with_llvm_err = failed to prepare thin LTO context: {$llvm_err}

codegen_llvm_load_bitcode = failed to load bitcode of module "{$name}"
codegen_llvm_load_bitcode_with_llvm_err = failed to load bitcode of module "{$name}": {$llvm_err}

codegen_llvm_write_thinlto_key = error while writing ThinLTO key data: {$err}
codegen_llvm_write_thinlto_key_with_llvm_err = error while writing ThinLTO key data: {$err}: {$llvm_err}

codegen_llvm_multiple_source_dicompileunit = multiple source DICompileUnits found
codegen_llvm_multiple_source_dicompileunit_with_llvm_err = multiple source DICompileUnits found: {$llvm_err}

codegen_llvm_prepare_thin_lto_module = failed to prepare thin LTO module
codegen_llvm_prepare_thin_lto_module_with_llvm_err = failed to prepare thin LTO module: {$llvm_err}

codegen_llvm_parse_bitcode = failed to parse bitcode for LTO module
codegen_llvm_parse_bitcode_with_llvm_err = failed to parse bitcode for LTO module: {$llvm_err}

codegen_llvm_from_llvm_optimization_diag = {$filename}:{$line}:{$column} {$pass_name} ({$kind}): {$message}
codegen_llvm_from_llvm_diag = {$message}

codegen_llvm_write_bytecode = failed to write bytecode to {$path}: {$err}

codegen_llvm_copy_bitcode = failed to copy bitcode to object file: {$err}
