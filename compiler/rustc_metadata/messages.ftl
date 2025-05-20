metadata_as_needed_compatibility =
    linking modifier `as-needed` is only compatible with `dylib` and `framework` linking kinds

metadata_async_drop_types_in_dependency =
    found async drop types in dependency `{$extern_crate}`, but async_drop feature is disabled for `{$local_crate}`
    .help = if async drop type will be dropped in a crate without `feature(async_drop)`, sync Drop will be used

metadata_bad_panic_strategy =
    the linked panic runtime `{$runtime}` is not compiled with this crate's panic strategy `{$strategy}`

metadata_binary_output_to_tty =
    option `-o` or `--emit` is used to write binary output type `metadata` to stdout, but stdout is a tty

metadata_bundle_needs_static =
    linking modifier `bundle` is only compatible with `static` linking kind

metadata_cannot_find_crate =
    can't find crate for `{$crate_name}`{$add_info}

metadata_cant_find_crate =
    can't find crate

metadata_compiler_missing_profiler =
    the compiler may have been built without the profiler runtime

metadata_conflicting_alloc_error_handler =
    the `#[alloc_error_handler]` in {$other_crate_name} conflicts with allocation error handler in: {$crate_name}

metadata_conflicting_global_alloc =
    the `#[global_allocator]` in {$other_crate_name} conflicts with global allocator in: {$crate_name}

metadata_consider_adding_std =
    consider adding the standard library to the sysroot with `x build library --target {$locator_triple}`

metadata_consider_building_std =
    consider building the standard library from source with `cargo build -Zbuild-std`

metadata_consider_downloading_target =
    consider downloading the target with `rustup target add {$locator_triple}`

metadata_crate_dep_multiple =
    cannot satisfy dependencies so `{$crate_name}` only shows up once
    .help = having upstream crates all available in one format will likely make this go away

metadata_crate_dep_not_static =
    `{$crate_name}` was unavailable as a static crate, preventing fully static linking

metadata_crate_dep_rustc_driver =
    `feature(rustc_private)` is needed to link to the compiler's `rustc_driver` library

metadata_crate_location_unknown_type =
    extern location for {$crate_name} is of an unknown type: {$path}

metadata_crate_not_compiler_builtins =
    the crate `{$crate_name}` resolved as `compiler_builtins` but is not `#![compiler_builtins]`

metadata_crate_not_panic_runtime =
    the crate `{$crate_name}` is not a panic runtime

metadata_dl_error =
    {$path}{$err}

metadata_empty_link_name =
    link name must not be empty
    .label = empty link name

metadata_empty_renaming_target =
    an empty renaming target was specified for library `{$lib_name}`

metadata_extern_location_not_exist =
    extern location for {$crate_name} does not exist: {$location}

metadata_extern_location_not_file =
    extern location for {$crate_name} is not a file: {$location}

metadata_fail_create_file_encoder =
    failed to create file encoder: {$err}

metadata_fail_write_file =
    failed to write to `{$path}`: {$err}

metadata_failed_copy_to_stdout =
    failed to copy {$filename} to stdout: {$err}

metadata_failed_create_encoded_metadata =
    failed to create encoded metadata from file: {$err}

metadata_failed_create_file =
    failed to create the file {$filename}: {$err}

metadata_failed_create_tempdir =
    couldn't create a temp dir: {$err}

metadata_failed_write_error =
    failed to write {$filename}: {$err}

metadata_found_crate_versions =
    the following crate versions were found:{$found_crates}

metadata_found_staticlib =
    found staticlib `{$crate_name}` instead of rlib or dylib{$add_info}
    .help = please recompile that crate using --crate-type lib

metadata_full_metadata_not_found =
    only metadata stub found for `{$flavor}` dependency `{$crate_name}`
    please provide path to the corresponding .rmeta file with full metadata

metadata_global_alloc_required =
    no global memory allocator found but one is required; link to std or add `#[global_allocator]` to a static item that implements the GlobalAlloc trait

metadata_import_name_type_form =
    import name type must be of the form `import_name_type = "string"`

metadata_import_name_type_raw =
    import name type can only be used with link kind `raw-dylib`

metadata_import_name_type_x86 =
    import name type is only supported on x86

metadata_incompatible_panic_in_drop_strategy =
    the crate `{$crate_name}` is compiled with the panic-in-drop strategy `{$found_strategy}` which is incompatible with this crate's strategy of `{$desired_strategy}`

metadata_incompatible_rustc =
    found crate `{$crate_name}` compiled by an incompatible version of rustc{$add_info}
    .help = please recompile that crate using this compiler ({$rustc_version}) (consider running `cargo clean` first)

metadata_incompatible_target_modifiers =
    mixing `{$flag_name_prefixed}` will cause an ABI mismatch in crate `{$local_crate}`
    .note = `{$flag_name_prefixed}={$local_value}` in this crate is incompatible with `{$flag_name_prefixed}={$extern_value}` in dependency `{$extern_crate}`
    .help = the `{$flag_name_prefixed}` flag modifies the ABI so Rust crates compiled with different values of this flag cannot be used together safely
metadata_incompatible_target_modifiers_help_allow = if you are sure this will not cause problems, you may use `-Cunsafe-allow-abi-mismatch={$flag_name}` to silence this error
metadata_incompatible_target_modifiers_help_fix = set `{$flag_name_prefixed}={$extern_value}` in this crate or `{$flag_name_prefixed}={$local_value}` in `{$extern_crate}`

metadata_incompatible_target_modifiers_help_fix_l_missed = set `{$flag_name_prefixed}={$extern_value}` in this crate or unset `{$flag_name_prefixed}` in `{$extern_crate}`

metadata_incompatible_target_modifiers_help_fix_r_missed = unset `{$flag_name_prefixed}` in this crate or set `{$flag_name_prefixed}={$local_value}` in `{$extern_crate}`

metadata_incompatible_target_modifiers_l_missed =
    mixing `{$flag_name_prefixed}` will cause an ABI mismatch in crate `{$local_crate}`
    .note = unset `{$flag_name_prefixed}` in this crate is incompatible with `{$flag_name_prefixed}={$extern_value}` in dependency `{$extern_crate}`
    .help = the `{$flag_name_prefixed}` flag modifies the ABI so Rust crates compiled with different values of this flag cannot be used together safely
metadata_incompatible_target_modifiers_r_missed =
    mixing `{$flag_name_prefixed}` will cause an ABI mismatch in crate `{$local_crate}`
    .note = `{$flag_name_prefixed}={$local_value}` in this crate is incompatible with unset `{$flag_name_prefixed}` in dependency `{$extern_crate}`
    .help = the `{$flag_name_prefixed}` flag modifies the ABI so Rust crates compiled with different values of this flag cannot be used together safely
metadata_incompatible_wasm_link =
    `wasm_import_module` is incompatible with other arguments in `#[link]` attributes

metadata_install_missing_components =
    maybe you need to install the missing components with: `rustup component add rust-src rustc-dev llvm-tools-preview`

metadata_invalid_link_modifier =
    invalid linking modifier syntax, expected '+' or '-' prefix before one of: bundle, verbatim, whole-archive, as-needed

metadata_invalid_meta_files =
    found invalid metadata files for crate `{$crate_name}`{$add_info}

metadata_lib_filename_form =
    file name should be lib*.rlib or {$dll_prefix}*{$dll_suffix}

metadata_lib_framework_apple =
    library kind `framework` is only supported on Apple targets

metadata_lib_required =
    crate `{$crate_name}` required to be available in {$kind} format, but was not found in this form

metadata_link_arg_unstable =
    link kind `link-arg` is unstable

metadata_link_cfg_form =
    link cfg must be of the form `cfg(/* predicate */)`

metadata_link_cfg_single_predicate =
    link cfg must have a single predicate argument

metadata_link_cfg_unstable =
    link cfg is unstable

metadata_link_framework_apple =
    link kind `framework` is only supported on Apple targets

metadata_link_kind_form =
    link kind must be of the form `kind = "string"`

metadata_link_modifiers_form =
    link modifiers must be of the form `modifiers = "string"`

metadata_link_name_form =
    link name must be of the form `name = "string"`

metadata_link_ordinal_raw_dylib =
    `#[link_ordinal]` is only supported if link kind is `raw-dylib`

metadata_link_requires_name =
    `#[link]` attribute requires a `name = "string"` argument
    .label = missing `name` argument

metadata_missing_native_library =
    could not find native static library `{$libname}`, perhaps an -L flag is missing?

metadata_multiple_candidates =
    multiple candidates for `{$flavor}` dependency `{$crate_name}` found

metadata_multiple_cfgs =
    multiple `cfg` arguments in a single `#[link]` attribute

metadata_multiple_import_name_type =
    multiple `import_name_type` arguments in a single `#[link]` attribute

metadata_multiple_kinds_in_link =
    multiple `kind` arguments in a single `#[link]` attribute

metadata_multiple_link_modifiers =
    multiple `modifiers` arguments in a single `#[link]` attribute

metadata_multiple_modifiers =
    multiple `{$modifier}` modifiers in a single `modifiers` argument

metadata_multiple_names_in_link =
    multiple `name` arguments in a single `#[link]` attribute

metadata_multiple_renamings =
    multiple renamings were specified for library `{$lib_name}`

metadata_multiple_wasm_import =
    multiple `wasm_import_module` arguments in a single `#[link]` attribute

metadata_newer_crate_version =
    found possibly newer version of crate `{$crate_name}`{$add_info}
    .note = perhaps that crate needs to be recompiled?

metadata_no_crate_with_triple =
    couldn't find crate `{$crate_name}` with expected target triple {$locator_triple}{$add_info}

metadata_no_link_mod_override =
    overriding linking modifiers from command line is not supported

metadata_no_multiple_alloc_error_handler =
    cannot define multiple allocation error handlers
    .label = cannot define a new allocation error handler

metadata_no_multiple_global_alloc =
    cannot define multiple global allocators
    .label = cannot define a new global allocator

metadata_no_panic_strategy =
    the crate `{$crate_name}` does not have the panic strategy `{$strategy}`

metadata_no_transitive_needs_dep =
    the crate `{$crate_name}` cannot depend on a crate that needs {$needs_crate_name}, but it depends on `{$deps_crate_name}`

metadata_non_ascii_name =
    cannot load a crate with a non-ascii name `{$crate_name}`

metadata_not_profiler_runtime =
    the crate `{$crate_name}` is not a profiler runtime

metadata_only_provide_library_name = only provide the library name `{$suggested_name}`, not the full filename

metadata_prev_alloc_error_handler =
    previous allocation error handler defined here

metadata_prev_global_alloc =
    previous global allocator defined here

metadata_raw_dylib_elf_unstable =
    link kind `raw-dylib` is unstable on ELF platforms

metadata_raw_dylib_no_nul =
    link name must not contain NUL characters if link kind is `raw-dylib`

metadata_raw_dylib_only_windows =
    link kind `raw-dylib` is only supported on Windows targets

metadata_renaming_no_link =
    renaming of the library `{$lib_name}` was specified, however this crate contains no `#[link(...)]` attributes referencing this library

metadata_required_panic_strategy =
    the crate `{$crate_name}` requires panic strategy `{$found_strategy}` which is incompatible with this crate's strategy of `{$desired_strategy}`

metadata_rlib_required =
    crate `{$crate_name}` required to be available in rlib format, but was not found in this form

metadata_rustc_lib_required =
    crate `{$crate_name}` required to be available in {$kind} format, but was not found in this form
    .note = only .rmeta files are distributed for `rustc_private` crates other than `rustc_driver`
    .help = try adding `extern crate rustc_driver;` at the top level of this crate

metadata_stable_crate_id_collision =
    found crates (`{$crate_name0}` and `{$crate_name1}`) with colliding StableCrateId values

metadata_std_required =
    `std` is required by `{$current_crate}` because it does not declare `#![no_std]`

metadata_symbol_conflicts_current =
    the current crate is indistinguishable from one of its dependencies: it has the same crate-name `{$crate_name}` and was compiled with the same `-C metadata` arguments, so this will result in symbol conflicts between the two

metadata_target_no_std_support =
    the `{$locator_triple}` target may not support the standard library

metadata_target_not_installed =
    the `{$locator_triple}` target may not be installed

metadata_two_panic_runtimes =
    cannot link together two panic runtimes: {$prev_name} and {$cur_name}

metadata_unexpected_link_arg =
    unexpected `#[link]` argument, expected one of: name, kind, modifiers, cfg, wasm_import_module, import_name_type

metadata_unknown_import_name_type =
    unknown import name type `{$import_name_type}`, expected one of: decorated, noprefix, undecorated

metadata_unknown_link_kind =
    unknown link kind `{$kind}`, expected one of: static, dylib, framework, raw-dylib, link-arg
    .label = unknown link kind

metadata_unknown_link_modifier =
    unknown linking modifier `{$modifier}`, expected one of: bundle, verbatim, whole-archive, as-needed

metadata_unknown_target_modifier_unsafe_allowed = unknown target modifier `{$flag_name}`, requested by `-Cunsafe-allow-abi-mismatch={$flag_name}`

metadata_unsupported_abi =
    ABI not supported by `#[link(kind = "raw-dylib")]` on this architecture

metadata_unsupported_abi_i686 =
    ABI not supported by `#[link(kind = "raw-dylib")]` on i686

metadata_wasm_c_abi =
    older versions of the `wasm-bindgen` crate are incompatible with current versions of Rust; please update to `wasm-bindgen` v0.2.88

metadata_wasm_import_form =
    wasm import module must be of the form `wasm_import_module = "string"`

metadata_whole_archive_needs_static =
    linking modifier `whole-archive` is only compatible with `static` linking kind
