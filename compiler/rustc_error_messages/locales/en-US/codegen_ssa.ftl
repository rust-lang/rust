codegen_ssa_lib_def_write_failure = failed to write lib.def file: {$error}

codegen_ssa_version_script_write_failure = failed to write version script: {$error}

codegen_ssa_symbol_file_write_failure = failed to write symbols file: {$error}

codegen_ssa_unsupported_arch = arch is not supported

codegen_ssa_msvc_path_not_found = MSVC root path lib location not found

codegen_ssa_link_exe_not_found = link.exe not found

codegen_ssa_ld64_unimplemented_modifier = `as-needed` modifier not implemented yet for ld64

codegen_ssa_linker_unsupported_modifier = `as-needed` modifier not supported for current linker

codegen_ssa_L4Bender_exporting_symbols_unimplemented = exporting symbols not implemented yet for L4Bender

codegen_ssa_no_natvis_directory = error enumerating natvis directory: {$error}

codegen_ssa_copy_path = could not copy {$from} to {$to}: {$error}

codegen_ssa_copy_path_buf = unable to copy {$source_file} to {$output_path}: {$error}

codegen_ssa_ignoring_emit_path = ignoring emit path because multiple .{$extension} files were produced

codegen_ssa_ignoring_output = ignoring -o because multiple .{$extension} files were produced

codegen_ssa_create_temp_dir = couldn't create a temp dir: {$error}

codegen_ssa_incompatible_linking_modifiers = the linking modifiers `+bundle` and `+whole-archive` are not compatible with each other when generating rlibs

codegen_ssa_add_native_library = failed to add native library {$library_path}: {$error}

codegen_ssa_multiple_external_func_decl = multiple declarations of external function `{$function}` from library `{$library_name}` have different calling conventions

codegen_ssa_rlib_missing_format = could not find formats for rlibs

codegen_ssa_rlib_only_rmeta_found = could not find rlib for: `{$crate_name}`, found rmeta (metadata) file

codegen_ssa_rlib_not_found = could not find rlib for: `{$crate_name}`

codegen_ssa_thorin_dwarf_linking = linking dwarf objects with thorin failed
    .note = {$thorin_error}

codegen_ssa_linking_failed = linking with `{$linker_path}` failed: {$exit_status}

codegen_ssa_extern_funcs_not_found = some `extern` functions couldn't be found; some native libraries may need to be installed or have their path specified

codegen_ssa_specify_libraries_to_link = use the `-l` flag to specify native libraries to link

codegen_ssa_use_cargo_directive = use the `cargo:rustc-link-lib` directive to specify the native libraries to link with Cargo (see https://doc.rust-lang.org/cargo/reference/build-scripts.html#cargorustc-link-libkindname)
