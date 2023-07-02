codegen_ssa_L4Bender_exporting_symbols_unimplemented = exporting symbols not implemented yet for L4Bender

codegen_ssa_add_native_library = failed to add native library {$library_path}: {$error}

codegen_ssa_apple_sdk_error_sdk_path = failed to get {$sdk_name} SDK path: {$error}

codegen_ssa_archive_build_failure =
    failed to build archive: {$error}

codegen_ssa_atomic_compare_exchange = Atomic compare-exchange intrinsic missing failure memory ordering

codegen_ssa_binary_output_to_tty = option `-o` or `--emit` is used to write binary output type `{$shorthand}` to stdout, but stdout is a tty

codegen_ssa_check_installed_visual_studio = please ensure that Visual Studio 2017 or later, or Build Tools for Visual Studio were installed with the Visual C++ option.

codegen_ssa_copy_path = could not copy {$from} to {$to}: {$error}

codegen_ssa_copy_path_buf = unable to copy {$source_file} to {$output_path}: {$error}

codegen_ssa_create_temp_dir = couldn't create a temp dir: {$error}

codegen_ssa_erroneous_constant = erroneous constant encountered

codegen_ssa_error_creating_remark_dir = failed to create remark directory: {$error}

codegen_ssa_expected_used_symbol = expected `used`, `used(compiler)` or `used(linker)`

codegen_ssa_extern_funcs_not_found = some `extern` functions couldn't be found; some native libraries may need to be installed or have their path specified

codegen_ssa_extract_bundled_libs_archive_member = failed to get data from archive member '{$rlib}': {$error}
codegen_ssa_extract_bundled_libs_convert_name = failed to convert name '{$rlib}': {$error}
codegen_ssa_extract_bundled_libs_mmap_file = failed to mmap file '{$rlib}': {$error}
codegen_ssa_extract_bundled_libs_open_file = failed to open file '{$rlib}': {$error}
codegen_ssa_extract_bundled_libs_parse_archive = failed to parse archive '{$rlib}': {$error}
codegen_ssa_extract_bundled_libs_read_entry = failed to read entry '{$rlib}': {$error}
codegen_ssa_extract_bundled_libs_write_file = failed to write file '{$rlib}': {$error}

codegen_ssa_failed_to_write = failed to write {$path}: {$error}

codegen_ssa_ignoring_emit_path = ignoring emit path because multiple .{$extension} files were produced

codegen_ssa_ignoring_output = ignoring -o because multiple .{$extension} files were produced

codegen_ssa_illegal_link_ordinal_format = illegal ordinal format in `link_ordinal`
    .note = an unsuffixed integer value, e.g., `1`, is expected

codegen_ssa_incompatible_linking_modifiers = link modifiers combination `+bundle,+whole-archive` is unstable when generating rlibs

codegen_ssa_insufficient_vs_code_product = VS Code is a different product, and is not sufficient.

codegen_ssa_invalid_link_ordinal_nargs = incorrect number of arguments to `#[link_ordinal]`
    .note = the attribute requires exactly one argument

codegen_ssa_invalid_monomorphization_basic_float_type = invalid monomorphization of `{$name}` intrinsic: expected basic float type, found `{$ty}`

codegen_ssa_invalid_monomorphization_basic_integer_type = invalid monomorphization of `{$name}` intrinsic: expected basic integer type, found `{$ty}`

codegen_ssa_invalid_monomorphization_cannot_return = invalid monomorphization of `{$name}` intrinsic: cannot return `{$ret_ty}`, expected `u{$expected_int_bits}` or `[u8; {$expected_bytes}]`

codegen_ssa_invalid_monomorphization_cast_fat_pointer = invalid monomorphization of `{$name}` intrinsic: cannot cast fat pointer `{$ty}`

codegen_ssa_invalid_monomorphization_expected_element_type = invalid monomorphization of `{$name}` intrinsic: expected element type `{$expected_element}` of second argument `{$second_arg}` to be a pointer to the element type `{$in_elem}` of the first argument `{$in_ty}`, found `{$expected_element}` != `{$mutability} {$in_elem}`

codegen_ssa_invalid_monomorphization_expected_pointer = invalid monomorphization of `{$name}` intrinsic: expected pointer, got `{$ty}`

codegen_ssa_invalid_monomorphization_expected_return_type = invalid monomorphization of `{$name}` intrinsic: expected return type `{$in_ty}`, found `{$ret_ty}`

codegen_ssa_invalid_monomorphization_expected_usize = invalid monomorphization of `{$name}` intrinsic: expected `usize`, got `{$ty}`

codegen_ssa_invalid_monomorphization_expected_vector_element_type = invalid monomorphization of `{$name}` intrinsic: expected element type `{$expected_element}` of vector type `{$vector_type}` to be a signed or unsigned integer type

codegen_ssa_invalid_monomorphization_float_to_int_unchecked = invalid monomorphization of `float_to_int_unchecked` intrinsic: expected basic float type, found `{$ty}`

codegen_ssa_invalid_monomorphization_floating_point_type = invalid monomorphization of `{$name}` intrinsic: `{$in_ty}` is not a floating-point type

codegen_ssa_invalid_monomorphization_floating_point_vector = invalid monomorphization of `{$name}` intrinsic: unsupported element type `{$f_ty}` of floating-point vector `{$in_ty}`

codegen_ssa_invalid_monomorphization_inserted_type = invalid monomorphization of `{$name}` intrinsic: expected inserted type `{$in_elem}` (element of input `{$in_ty}`), found `{$out_ty}`

codegen_ssa_invalid_monomorphization_invalid_bitmask = invalid monomorphization of `{$name}` intrinsic: invalid bitmask `{$mask_ty}`, expected `u{$expected_int_bits}` or `[u8; {$expected_bytes}]`

codegen_ssa_invalid_monomorphization_mask_type = invalid monomorphization of `{$name}` intrinsic: mask element type is `{$ty}`, expected `i_`

codegen_ssa_invalid_monomorphization_mismatched_lengths = invalid monomorphization of `{$name}` intrinsic: mismatched lengths: mask length `{$m_len}` != other vector length `{$v_len}`

codegen_ssa_invalid_monomorphization_return_element = invalid monomorphization of `{$name}` intrinsic: expected return element type `{$in_elem}` (element of input `{$in_ty}`), found `{$ret_ty}` with element type `{$out_ty}`

codegen_ssa_invalid_monomorphization_return_integer_type = invalid monomorphization of `{$name}` intrinsic: expected return type with integer elements, found `{$ret_ty}` with non-integer `{$out_ty}`

codegen_ssa_invalid_monomorphization_return_length = invalid monomorphization of `{$name}` intrinsic: expected return type of length {$in_len}, found `{$ret_ty}` with length {$out_len}

codegen_ssa_invalid_monomorphization_return_length_input_type = invalid monomorphization of `{$name}` intrinsic: expected return type with length {$in_len} (same as input type `{$in_ty}`), found `{$ret_ty}` with length {$out_len}

codegen_ssa_invalid_monomorphization_return_type = invalid monomorphization of `{$name}` intrinsic: expected return type `{$in_elem}` (element of input `{$in_ty}`), found `{$ret_ty}`

codegen_ssa_invalid_monomorphization_second_argument_length = invalid monomorphization of `{$name}` intrinsic: expected second argument with length {$in_len} (same as input type `{$in_ty}`), found `{$arg_ty}` with length {$out_len}

codegen_ssa_invalid_monomorphization_shuffle_index_not_constant = invalid monomorphization of `{$name}` intrinsic: shuffle index #{$arg_idx} is not a constant

codegen_ssa_invalid_monomorphization_shuffle_index_out_of_bounds = invalid monomorphization of `{$name}` intrinsic: shuffle index #{$arg_idx} is out of bounds (limit {$total_len})

codegen_ssa_invalid_monomorphization_simd_argument = invalid monomorphization of `{$name}` intrinsic: expected SIMD argument type, found non-SIMD `{$ty}`

codegen_ssa_invalid_monomorphization_simd_first = invalid monomorphization of `{$name}` intrinsic: expected SIMD first type, found non-SIMD `{$ty}`

codegen_ssa_invalid_monomorphization_simd_input = invalid monomorphization of `{$name}` intrinsic: expected SIMD input type, found non-SIMD `{$ty}`

codegen_ssa_invalid_monomorphization_simd_return = invalid monomorphization of `{$name}` intrinsic: expected SIMD return type, found non-SIMD `{$ty}`

codegen_ssa_invalid_monomorphization_simd_second = invalid monomorphization of `{$name}` intrinsic: expected SIMD second type, found non-SIMD `{$ty}`

codegen_ssa_invalid_monomorphization_simd_shuffle = invalid monomorphization of `{$name}` intrinsic: simd_shuffle index must be an array of `u32`, got `{$ty}`

codegen_ssa_invalid_monomorphization_simd_third = invalid monomorphization of `{$name}` intrinsic: expected SIMD third type, found non-SIMD `{$ty}`

codegen_ssa_invalid_monomorphization_third_arg_element_type = invalid monomorphization of `{$name}` intrinsic: expected element type `{$expected_element}` of third argument `{$third_arg}` to be a signed integer type

codegen_ssa_invalid_monomorphization_third_argument_length = invalid monomorphization of `{$name}` intrinsic: expected third argument with length {$in_len} (same as input type `{$in_ty}`), found `{$arg_ty}` with length {$out_len}

codegen_ssa_invalid_monomorphization_unrecognized_intrinsic = invalid monomorphization of `{$name}` intrinsic: unrecognized intrinsic `{$name}`

codegen_ssa_invalid_monomorphization_unsupported_cast = invalid monomorphization of `{$name}` intrinsic: unsupported cast from `{$in_ty}` with element `{$in_elem}` to `{$ret_ty}` with element `{$out_elem}`

codegen_ssa_invalid_monomorphization_unsupported_operation = invalid monomorphization of `{$name}` intrinsic: unsupported operation on `{$in_ty}` with element `{$in_elem}`

codegen_ssa_invalid_monomorphization_unsupported_symbol = invalid monomorphization of `{$name}` intrinsic: unsupported {$symbol} from `{$in_ty}` with element `{$in_elem}` to `{$ret_ty}`

codegen_ssa_invalid_monomorphization_unsupported_symbol_of_size = invalid monomorphization of `{$name}` intrinsic: unsupported {$symbol} from `{$in_ty}` with element `{$in_elem}` of size `{$size}` to `{$ret_ty}`

codegen_ssa_invalid_monomorphization_vector_argument = invalid monomorphization of `{$name}` intrinsic: vector argument `{$in_ty}`'s element type `{$in_elem}`, expected integer element type

codegen_ssa_invalid_no_sanitize = invalid argument for `no_sanitize`
    .note = expected one of: `address`, `cfi`, `hwaddress`, `kcfi`, `memory`, `memtag`, `shadow-call-stack`, or `thread`

codegen_ssa_invalid_windows_subsystem = invalid windows subsystem `{$subsystem}`, only `windows` and `console` are allowed

codegen_ssa_ld64_unimplemented_modifier = `as-needed` modifier not implemented yet for ld64

codegen_ssa_lib_def_write_failure = failed to write lib.def file: {$error}

codegen_ssa_link_exe_unexpected_error = `link.exe` returned an unexpected error

codegen_ssa_link_script_unavailable = can only use link script when linking with GNU-like linker

codegen_ssa_link_script_write_failure = failed to write link script to {$path}: {$error}

codegen_ssa_linker_file_stem = couldn't extract file stem from specified linker

codegen_ssa_linker_not_found = linker `{$linker_path}` not found
    .note = {$error}

codegen_ssa_linker_unsupported_modifier = `as-needed` modifier not supported for current linker

codegen_ssa_linking_failed = linking with `{$linker_path}` failed: {$exit_status}

codegen_ssa_metadata_object_file_write = error writing metadata object file: {$error}

codegen_ssa_missing_cpp_build_tool_component = or a necessary component may be missing from the "C++ build tools" workload

codegen_ssa_missing_memory_ordering = Atomic intrinsic missing memory ordering

codegen_ssa_msvc_missing_linker = the msvc targets depend on the msvc linker but `link.exe` was not found

codegen_ssa_multiple_external_func_decl = multiple declarations of external function `{$function}` from library `{$library_name}` have different calling conventions

codegen_ssa_multiple_main_functions = entry symbol `main` declared multiple times
    .help = did you use `#[no_mangle]` on `fn main`? Use `#[start]` instead

codegen_ssa_no_natvis_directory = error enumerating natvis directory: {$error}

codegen_ssa_option_gcc_only = option `-Z gcc-ld` is used even though linker flavor is not gcc

codegen_ssa_polymorphic_constant_too_generic = codegen encountered polymorphic constant: TooGeneric

codegen_ssa_processing_dymutil_failed = processing debug info with `dsymutil` failed: {$status}
    .note = {$output}

codegen_ssa_read_file = failed to read file: {$message}

codegen_ssa_repair_vs_build_tools = the Visual Studio build tools may need to be repaired using the Visual Studio installer

codegen_ssa_rlib_archive_build_failure = failed to build archive from rlib: {$error}

codegen_ssa_rlib_incompatible_dependency_formats = `{$ty1}` and `{$ty2}` do not have equivalent dependency formats (`{$list1}` vs `{$list2}`)

codegen_ssa_rlib_missing_format = could not find formats for rlibs

codegen_ssa_rlib_not_found = could not find rlib for: `{$crate_name}`

codegen_ssa_rlib_only_rmeta_found = could not find rlib for: `{$crate_name}`, found rmeta (metadata) file

codegen_ssa_select_cpp_build_tool_workload = in the Visual Studio installer, ensure the "C++ build tools" workload is selected

codegen_ssa_shuffle_indices_evaluation = could not evaluate shuffle_indices at compile time

codegen_ssa_specify_libraries_to_link = use the `-l` flag to specify native libraries to link

codegen_ssa_static_library_native_artifacts = Link against the following native artifacts when linking against this static library. The order and any duplication can be significant on some platforms.

codegen_ssa_stripping_debug_info_failed = stripping debug info with `{$util}` failed: {$status}
    .note = {$output}

codegen_ssa_symbol_file_write_failure = failed to write symbols file: {$error}

codegen_ssa_target_feature_safe_trait = `#[target_feature(..)]` cannot be applied to safe trait method
    .label = cannot be applied to safe trait method
    .label_def = not an `unsafe` function

codegen_ssa_thorin_decompress_data = failed to decompress compressed section

codegen_ssa_thorin_duplicate_unit = duplicate split compilation unit ({$unit})

codegen_ssa_thorin_empty_unit = unit {$unit} in input DWARF object with no data

codegen_ssa_thorin_gimli_read = {$error}
codegen_ssa_thorin_gimli_write = {$error}

codegen_ssa_thorin_incompatible_index_version = incompatible `{$section}` index version: found version {$actual}, expected version {$format}

codegen_ssa_thorin_invalid_input_kind = input is not an archive or elf object

codegen_ssa_thorin_io = {$error}
codegen_ssa_thorin_missing_dwo_name = missing path attribute to DWARF object ({$id})

codegen_ssa_thorin_missing_referenced_unit = unit {$unit} referenced by executable was not found

codegen_ssa_thorin_missing_required_section = input object missing required section `{$section}`

codegen_ssa_thorin_mixed_input_encodings = input objects haved mixed encodings

codegen_ssa_thorin_multiple_debug_info_section = multiple `.debug_info.dwo` sections

codegen_ssa_thorin_multiple_debug_types_section = multiple `.debug_types.dwo` sections in a package

codegen_ssa_thorin_multiple_relocations = multiple relocations for section `{$section}` at offset {$offset}

codegen_ssa_thorin_no_compilation_units = input object has no compilation units

codegen_ssa_thorin_no_die = no top-level debugging information entry in compilation/type unit

codegen_ssa_thorin_not_output_object_created = no output object was created from inputs

codegen_ssa_thorin_not_split_unit = regular compilation unit in object (missing dwo identifier)

codegen_ssa_thorin_object_read = {$error}
codegen_ssa_thorin_object_write = {$error}
codegen_ssa_thorin_offset_at_index = read offset at index {$index} of `.debug_str_offsets.dwo` section

codegen_ssa_thorin_parse_archive_member = failed to parse archive member

codegen_ssa_thorin_parse_index = failed to parse `{$section}` index section

codegen_ssa_thorin_parse_input_archive_file = failed to parse input archive file

codegen_ssa_thorin_parse_input_file_kind = failed to parse input file kind

codegen_ssa_thorin_parse_input_object_file = failed to parse input object file

codegen_ssa_thorin_parse_unit = failed to parse unit

codegen_ssa_thorin_parse_unit_abbreviations = failed to parse unit abbreviations

codegen_ssa_thorin_parse_unit_attribute = failed to parse unit attribute

codegen_ssa_thorin_parse_unit_header = failed to parse unit header

codegen_ssa_thorin_read_input_failure = failed to read input file

codegen_ssa_thorin_relocation_with_invalid_symbol = relocation with invalid symbol for section `{$section}` at offset {$offset}

codegen_ssa_thorin_row_not_in_index = row {$row} found in index's hash table not present in index

codegen_ssa_thorin_section_not_in_row = section not found in unit's row in index

codegen_ssa_thorin_section_without_name = section without name at offset {$offset}

codegen_ssa_thorin_str_at_offset = read string at offset {$offset} of `.debug_str.dwo` section

codegen_ssa_thorin_top_level_die_not_unit = top-level debugging information entry is not a compilation/type unit

codegen_ssa_thorin_unit_not_in_index = unit {$unit} from input package is not in its index

codegen_ssa_thorin_unsupported_relocation = unsupported relocation for section {$section} at offset {$offset}

codegen_ssa_unable_to_exe_linker = could not exec the linker `{$linker_path}`
    .note = {$error}
    .command_note = {$command_formatted}

codegen_ssa_unable_to_run = unable to run `{$util}`: {$error}

codegen_ssa_unable_to_run_dsymutil = unable to run `dsymutil`: {$error}

codegen_ssa_unable_to_write_debugger_visualizer = Unable to write debugger visualizer file `{$path}`: {$error}

codegen_ssa_unknown_archive_kind =
    Don't know how to build archive of type: {$kind}

codegen_ssa_unknown_atomic_operation = unknown atomic operation

codegen_ssa_unknown_atomic_ordering = unknown ordering in atomic intrinsic

codegen_ssa_unsupported_arch = unsupported arch `{$arch}` for os `{$os}`

codegen_ssa_unsupported_link_self_contained = option `-C link-self-contained` is not supported on this target

codegen_ssa_use_cargo_directive = use the `cargo:rustc-link-lib` directive to specify the native libraries to link with Cargo (see https://doc.rust-lang.org/cargo/reference/build-scripts.html#cargorustc-link-libkindname)

codegen_ssa_version_script_write_failure = failed to write version script: {$error}

codegen_ssa_visual_studio_not_installed = you may need to install Visual Studio build tools with the "C++ build tools" workload
