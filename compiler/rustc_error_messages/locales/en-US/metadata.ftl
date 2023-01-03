metadata_rlib_required =
    crate `{$crate_name}` required to be available in rlib format, but was not found in this form

metadata_lib_required =
    crate `{$crate_name}` required to be available in {$kind} format, but was not found in this form

metadata_rustc_lib_required =
    crate `{$crate_name}` required to be available in {$kind} format, but was not found in this form
    .note = only .rmeta files are distributed for `rustc_private` crates other than `rustc_driver`
    .help = try adding `extern crate rustc_driver;` at the top level of this crate

metadata_crate_dep_multiple =
    cannot satisfy dependencies so `{$crate_name}` only shows up once
    .help = having upstream crates all available in one format will likely make this go away

metadata_two_panic_runtimes =
    cannot link together two panic runtimes: {$prev_name} and {$cur_name}

metadata_bad_panic_strategy =
    the linked panic runtime `{$runtime}` is not compiled with this crate's panic strategy `{$strategy}`

metadata_required_panic_strategy =
    the crate `{$crate_name}` requires panic strategy `{$found_strategy}` which is incompatible with this crate's strategy of `{$desired_strategy}`

metadata_incompatible_panic_in_drop_strategy =
    the crate `{$crate_name}` is compiled with the panic-in-drop strategy `{$found_strategy}` which is incompatible with this crate's strategy of `{$desired_strategy}`

metadata_multiple_names_in_link =
    multiple `name` arguments in a single `#[link]` attribute

metadata_multiple_kinds_in_link =
    multiple `kind` arguments in a single `#[link]` attribute

metadata_link_name_form =
    link name must be of the form `name = "string"`

metadata_link_kind_form =
    link kind must be of the form `kind = "string"`

metadata_link_modifiers_form =
    link modifiers must be of the form `modifiers = "string"`

metadata_link_cfg_form =
    link cfg must be of the form `cfg(/* predicate */)`

metadata_wasm_import_form =
    wasm import module must be of the form `wasm_import_module = "string"`

metadata_empty_link_name =
    link name must not be empty
    .label = empty link name

metadata_link_framework_apple =
    link kind `framework` is only supported on Apple targets

metadata_framework_only_windows =
    link kind `raw-dylib` is only supported on Windows targets

metadata_unknown_link_kind =
    unknown link kind `{$kind}`, expected one of: static, dylib, framework, raw-dylib
    .label = unknown link kind

metadata_multiple_link_modifiers =
    multiple `modifiers` arguments in a single `#[link]` attribute

metadata_multiple_cfgs =
    multiple `cfg` arguments in a single `#[link]` attribute

metadata_link_cfg_single_predicate =
    link cfg must have a single predicate argument

metadata_multiple_wasm_import =
    multiple `wasm_import_module` arguments in a single `#[link]` attribute

metadata_unexpected_link_arg =
    unexpected `#[link]` argument, expected one of: name, kind, modifiers, cfg, wasm_import_module, import_name_type

metadata_invalid_link_modifier =
    invalid linking modifier syntax, expected '+' or '-' prefix before one of: bundle, verbatim, whole-archive, as-needed

metadata_multiple_modifiers =
    multiple `{$modifier}` modifiers in a single `modifiers` argument

metadata_bundle_needs_static =
    linking modifier `bundle` is only compatible with `static` linking kind

metadata_whole_archive_needs_static =
    linking modifier `whole-archive` is only compatible with `static` linking kind

metadata_as_needed_compatibility =
    linking modifier `as-needed` is only compatible with `dylib` and `framework` linking kinds

metadata_unknown_link_modifier =
    unknown linking modifier `{$modifier}`, expected one of: bundle, verbatim, whole-archive, as-needed

metadata_incompatible_wasm_link =
    `wasm_import_module` is incompatible with other arguments in `#[link]` attributes

metadata_link_requires_name =
    `#[link]` attribute requires a `name = "string"` argument
    .label = missing `name` argument

metadata_raw_dylib_no_nul =
    link name must not contain NUL characters if link kind is `raw-dylib`

metadata_link_ordinal_raw_dylib =
    `#[link_ordinal]` is only supported if link kind is `raw-dylib`

metadata_lib_framework_apple =
    library kind `framework` is only supported on Apple targets

metadata_empty_renaming_target =
    an empty renaming target was specified for library `{$lib_name}`

metadata_renaming_no_link =
    renaming of the library `{$lib_name}` was specified, however this crate contains no `#[link(...)]` attributes referencing this library

metadata_multiple_renamings =
    multiple renamings were specified for library `{$lib_name}`

metadata_no_link_mod_override =
    overriding linking modifiers from command line is not supported

metadata_unsupported_abi_i686 =
    ABI not supported by `#[link(kind = "raw-dylib")]` on i686

metadata_unsupported_abi =
    ABI not supported by `#[link(kind = "raw-dylib")]` on this architecture

metadata_fail_create_file_encoder =
    failed to create file encoder: {$err}

metadata_fail_seek_file =
    failed to seek the file: {$err}

metadata_fail_write_file =
    failed to write to the file: {$err}

metadata_crate_not_panic_runtime =
    the crate `{$crate_name}` is not a panic runtime

metadata_no_panic_strategy =
    the crate `{$crate_name}` does not have the panic strategy `{$strategy}`

metadata_profiler_builtins_needs_core =
    `profiler_builtins` crate (required by compiler options) is not compatible with crate attribute `#![no_core]`

metadata_not_profiler_runtime =
    the crate `{$crate_name}` is not a profiler runtime

metadata_no_multiple_global_alloc =
    cannot define multiple global allocators
    .label = cannot define a new global allocator

metadata_prev_global_alloc =
    previous global allocator defined here

metadata_no_multiple_alloc_error_handler =
    cannot define multiple allocation error handlers
    .label = cannot define a new allocation error handler

metadata_prev_alloc_error_handler =
    previous allocation error handler defined here

metadata_conflicting_global_alloc =
    the `#[global_allocator]` in {$other_crate_name} conflicts with global allocator in: {$crate_name}

metadata_conflicting_alloc_error_handler =
    the `#[alloc_error_handler]` in {$other_crate_name} conflicts with allocation error handler in: {$crate_name}

metadata_global_alloc_required =
    no global memory allocator found but one is required; link to std or add `#[global_allocator]` to a static item that implements the GlobalAlloc trait

metadata_no_transitive_needs_dep =
    the crate `{$crate_name}` cannot depend on a crate that needs {$needs_crate_name}, but it depends on `{$deps_crate_name}`

metadata_failed_write_error =
    failed to write {$filename}: {$err}

metadata_missing_native_library =
    could not find native static library `{$libname}`, perhaps an -L flag is missing?

metadata_only_provide_library_name = only provide the library name `{$suggested_name}`, not the full filename

metadata_failed_create_tempdir =
    couldn't create a temp dir: {$err}

metadata_failed_create_file =
    failed to create the file {$filename}: {$err}

metadata_failed_create_encoded_metadata =
    failed to create encoded metadata from file: {$err}

metadata_non_ascii_name =
    cannot load a crate with a non-ascii name `{$crate_name}`

metadata_extern_location_not_exist =
    extern location for {$crate_name} does not exist: {$location}

metadata_extern_location_not_file =
    extern location for {$crate_name} is not a file: {$location}

metadata_multiple_candidates =
    multiple candidates for `{$flavor}` dependency `{$crate_name}` found

metadata_symbol_conflicts_current =
    the current crate is indistinguishable from one of its dependencies: it has the same crate-name `{$crate_name}` and was compiled with the same `-C metadata` arguments. This will result in symbol conflicts between the two.

metadata_symbol_conflicts_others =
    found two different crates with name `{$crate_name}` that are not distinguished by differing `-C metadata`. This will result in symbol conflicts between the two.

metadata_stable_crate_id_collision =
    found crates (`{$crate_name0}` and `{$crate_name1}`) with colliding StableCrateId values.

metadata_dl_error =
    {$err}

metadata_newer_crate_version =
    found possibly newer version of crate `{$crate_name}`{$add_info}
    .note = perhaps that crate needs to be recompiled?

metadata_found_crate_versions =
    the following crate versions were found:{$found_crates}

metadata_no_crate_with_triple =
    couldn't find crate `{$crate_name}` with expected target triple {$locator_triple}{$add_info}

metadata_found_staticlib =
    found staticlib `{$crate_name}` instead of rlib or dylib{$add_info}
    .help = please recompile that crate using --crate-type lib

metadata_incompatible_rustc =
    found crate `{$crate_name}` compiled by an incompatible version of rustc{$add_info}
    .help = please recompile that crate using this compiler ({$rustc_version}) (consider running `cargo clean` first)

metadata_invalid_meta_files =
    found invalid metadata files for crate `{$crate_name}`{$add_info}

metadata_cannot_find_crate =
    can't find crate for `{$crate_name}`{$add_info}

metadata_no_dylib_plugin =
    plugin `{$crate_name}` only found in rlib format, but must be available in dylib format

metadata_target_not_installed =
    the `{$locator_triple}` target may not be installed

metadata_target_no_std_support =
    the `{$locator_triple}` target may not support the standard library

metadata_consider_downloading_target =
    consider downloading the target with `rustup target add {$locator_triple}`

metadata_std_required =
    `std` is required by `{$current_crate}` because it does not declare `#![no_std]`

metadata_consider_building_std =
    consider building the standard library from source with `cargo build -Zbuild-std`

metadata_compiler_missing_profiler =
    the compiler may have been built without the profiler runtime

metadata_install_missing_components =
    maybe you need to install the missing components with: `rustup component add rust-src rustc-dev llvm-tools-preview`

metadata_cant_find_crate =
    can't find crate

metadata_crate_location_unknown_type =
    extern location for {$crate_name} is of an unknown type: {$path}

metadata_lib_filename_form =
    file name should be lib*.rlib or {$dll_prefix}*{$dll_suffix}

metadata_multiple_import_name_type =
    multiple `import_name_type` arguments in a single `#[link]` attribute

metadata_import_name_type_form =
    import name type must be of the form `import_name_type = "string"`

metadata_import_name_type_x86 =
    import name type is only supported on x86

metadata_unknown_import_name_type =
    unknown import name type `{$import_name_type}`, expected one of: decorated, noprefix, undecorated

metadata_import_name_type_raw =
    import name type can only be used with link kind `raw-dylib`
