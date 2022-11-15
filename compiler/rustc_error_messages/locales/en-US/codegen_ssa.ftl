codegen_ssa_lib_def_write_failure = failed to write lib.def file: {$error}

codegen_ssa_version_script_write_failure = failed to write version script: {$error}

codegen_ssa_symbol_file_write_failure = failed to write symbols file: {$error}

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

codegen_ssa_rlib_incompatible_dependency_formats = `{$ty1}` and `{$ty2}` do not have equivalent dependency formats (`{$list1}` vs `{$list2}`)

codegen_ssa_linking_failed = linking with `{$linker_path}` failed: {$exit_status}

codegen_ssa_extern_funcs_not_found = some `extern` functions couldn't be found; some native libraries may need to be installed or have their path specified

codegen_ssa_specify_libraries_to_link = use the `-l` flag to specify native libraries to link

codegen_ssa_use_cargo_directive = use the `cargo:rustc-link-lib` directive to specify the native libraries to link with Cargo (see https://doc.rust-lang.org/cargo/reference/build-scripts.html#cargorustc-link-libkindname)

codegen_ssa_thorin_read_input_failure = failed to read input file

codegen_ssa_thorin_parse_input_file_kind = failed to parse input file kind

codegen_ssa_thorin_parse_input_object_file = failed to parse input object file

codegen_ssa_thorin_parse_input_archive_file = failed to parse input archive file

codegen_ssa_thorin_parse_archive_member = failed to parse archive member

codegen_ssa_thorin_invalid_input_kind = input is not an archive or elf object

codegen_ssa_thorin_decompress_data = failed to decompress compressed section

codegen_ssa_thorin_section_without_name = section without name at offset {$offset}

codegen_ssa_thorin_relocation_with_invalid_symbol = relocation with invalid symbol for section `{$section}` at offset {$offset}

codegen_ssa_thorin_multiple_relocations = multiple relocations for section `{$section}` at offset {$offset}

codegen_ssa_thorin_unsupported_relocation = unsupported relocation for section {$section} at offset {$offset}

codegen_ssa_thorin_missing_dwo_name = missing path attribute to DWARF object ({$id})

codegen_ssa_thorin_no_compilation_units = input object has no compilation units

codegen_ssa_thorin_no_die = no top-level debugging information entry in compilation/type unit

codegen_ssa_thorin_top_level_die_not_unit = top-level debugging information entry is not a compilation/type unit

codegen_ssa_thorin_missing_required_section = input object missing required section `{$section}`

codegen_ssa_thorin_parse_unit_abbreviations = failed to parse unit abbreviations

codegen_ssa_thorin_parse_unit_attribute = failed to parse unit attribute

codegen_ssa_thorin_parse_unit_header = failed to parse unit header

codegen_ssa_thorin_parse_unit = failed to parse unit

codegen_ssa_thorin_incompatible_index_version = incompatible `{$section}` index version: found version {$actual}, expected version {$format}

codegen_ssa_thorin_offset_at_index = read offset at index {$index} of `.debug_str_offsets.dwo` section

codegen_ssa_thorin_str_at_offset = read string at offset {$offset} of `.debug_str.dwo` section

codegen_ssa_thorin_parse_index = failed to parse `{$section}` index section

codegen_ssa_thorin_unit_not_in_index = unit {$unit} from input package is not in its index

codegen_ssa_thorin_row_not_in_index = row {$row} found in index's hash table not present in index

codegen_ssa_thorin_section_not_in_row = section not found in unit's row in index

codegen_ssa_thorin_empty_unit = unit {$unit} in input DWARF object with no data

codegen_ssa_thorin_multiple_debug_info_section = multiple `.debug_info.dwo` sections

codegen_ssa_thorin_multiple_debug_types_section = multiple `.debug_types.dwo` sections in a package

codegen_ssa_thorin_not_split_unit = regular compilation unit in object (missing dwo identifier)

codegen_ssa_thorin_duplicate_unit = duplicate split compilation unit ({$unit})

codegen_ssa_thorin_missing_referenced_unit = unit {$unit} referenced by executable was not found

codegen_ssa_thorin_not_output_object_created = no output object was created from inputs

codegen_ssa_thorin_mixed_input_encodings = input objects haved mixed encodings

codegen_ssa_thorin_io = {$error}
codegen_ssa_thorin_object_read = {$error}
codegen_ssa_thorin_object_write = {$error}
codegen_ssa_thorin_gimli_read = {$error}
codegen_ssa_thorin_gimli_write = {$error}

codegen_ssa_link_exe_unexpected_error = `link.exe` returned an unexpected error

codegen_ssa_repair_vs_build_tools = the Visual Studio build tools may need to be repaired using the Visual Studio installer

codegen_ssa_missing_cpp_build_tool_component = or a necessary component may be missing from the "C++ build tools" workload

codegen_ssa_select_cpp_build_tool_workload = in the Visual Studio installer, ensure the "C++ build tools" workload is selected

codegen_ssa_visual_studio_not_installed = you may need to install Visual Studio build tools with the "C++ build tools" workload

codegen_ssa_linker_not_found = linker `{$linker_path}` not found
    .note = {$error}

codegen_ssa_unable_to_exe_linker = could not exec the linker `{$linker_path}`
    .note = {$error}
    .command_note = {$command_formatted}

codegen_ssa_msvc_missing_linker = the msvc targets depend on the msvc linker but `link.exe` was not found

codegen_ssa_check_installed_visual_studio = please ensure that Visual Studio 2017 or later, or Build Tools for Visual Studio were installed with the Visual C++ option.

codegen_ssa_unsufficient_vs_code_product = VS Code is a different product, and is not sufficient.

codegen_ssa_processing_dymutil_failed = processing debug info with `dsymutil` failed: {$status}
    .note = {$output}

codegen_ssa_unable_to_run_dsymutil = unable to run `dsymutil`: {$error}

codegen_ssa_stripping_debu_info_failed = stripping debug info with `{$util}` failed: {$status}
    .note = {$output}

codegen_ssa_unable_to_run = unable to run `{$util}`: {$error}

codegen_ssa_linker_file_stem = couldn't extract file stem from specified linker

codegen_ssa_static_library_native_artifacts = Link against the following native artifacts when linking against this static library. The order and any duplication can be significant on some platforms.

codegen_ssa_native_static_libs = native-static-libs: {$arguments}

codegen_ssa_link_script_unavailable = can only use link script when linking with GNU-like linker

codegen_ssa_link_script_write_failure = failed to write link script to {$path}: {$error}

codegen_ssa_failed_to_write = failed to write {$path}: {$error}

codegen_ssa_unable_to_write_debugger_visualizer = Unable to write debugger visualizer file `{$path}`: {$error}

codegen_ssa_rlib_archive_build_failure = failed to build archive from rlib: {$error}

codegen_ssa_option_gcc_only = option `-Z gcc-ld` is used even though linker flavor is not gcc

codegen_ssa_extract_bundled_libs_open_file = failed to open file '{$rlib}': {$error}
codegen_ssa_extract_bundled_libs_mmap_file = failed to mmap file '{$rlib}': {$error}
codegen_ssa_extract_bundled_libs_parse_archive = failed to parse archive '{$rlib}': {$error}
codegen_ssa_extract_bundled_libs_read_entry = failed to read entry '{$rlib}': {$error}
codegen_ssa_extract_bundled_libs_archive_member = failed to get data from archive member '{$rlib}': {$error}
codegen_ssa_extract_bundled_libs_convert_name = failed to convert name '{$rlib}': {$error}
codegen_ssa_extract_bundled_libs_write_file = failed to write file '{$rlib}': {$error}

codegen_ssa_unsupported_arch = unsupported arch `{$arch}` for os `{$os}`

codegen_ssa_apple_sdk_error_sdk_path = failed to get {$sdk_name} SDK path: {error}

codegen_ssa_read_file = failed to read file: {message}
