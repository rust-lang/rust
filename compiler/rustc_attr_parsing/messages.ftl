attr_parsing_cfg_predicate_identifier =
    `cfg` predicate key must be an identifier

attr_parsing_deprecated_item_suggestion =
    suggestions on deprecated items are unstable
    .help = add `#![feature(deprecated_suggestion)]` to the crate root
    .note = see #94785 for more details

attr_parsing_empty_confusables =
    expected at least one confusable name
attr_parsing_expected_one_cfg_pattern =
    expected 1 cfg-pattern

attr_parsing_expected_single_version_literal =
    expected single version literal

attr_parsing_expected_version_literal =
    expected a version literal

attr_parsing_expects_feature_list =
    `{$name}` expects a list of feature names

attr_parsing_expects_features =
    `{$name}` expects feature names

attr_parsing_ill_formed_attribute_input = {$num_suggestions ->
        [1] attribute must be of the form {$suggestions}
        *[other] valid forms for the attribute are {$suggestions}
    }

attr_parsing_incorrect_repr_format_align_one_arg =
    incorrect `repr(align)` attribute format: `align` takes exactly one argument in parentheses

attr_parsing_incorrect_repr_format_expect_literal_integer =
    incorrect `repr(align)` attribute format: `align` expects a literal integer as argument

attr_parsing_incorrect_repr_format_generic =
    incorrect `repr({$repr_arg})` attribute format
    .suggestion = use parentheses instead

attr_parsing_incorrect_repr_format_packed_expect_integer =
    incorrect `repr(packed)` attribute format: `packed` expects a literal integer as argument

attr_parsing_incorrect_repr_format_packed_one_or_zero_arg =
    incorrect `repr(packed)` attribute format: `packed` takes exactly one parenthesized argument, or no parentheses at all

attr_parsing_invalid_alignment_value =
    invalid alignment value: {$error_part}

attr_parsing_invalid_issue_string =
    `issue` must be a non-zero numeric string or "none"
    .must_not_be_zero = `issue` must not be "0", use "none" instead
    .empty = cannot parse integer from empty string
    .invalid_digit = invalid digit found in string
    .pos_overflow = number too large to fit in target type
    .neg_overflow = number too small to fit in target type

attr_parsing_invalid_predicate =
    invalid predicate `{$predicate}`

attr_parsing_invalid_repr_align_need_arg =
    invalid `repr(align)` attribute: `align` needs an argument
    .suggestion = supply an argument here

attr_parsing_invalid_repr_generic =
    invalid `repr({$repr_arg})` attribute: {$error_part}

attr_parsing_invalid_repr_hint_no_paren =
    invalid representation hint: `{$name}` does not take a parenthesized argument list

attr_parsing_invalid_repr_hint_no_value =
    invalid representation hint: `{$name}` does not take a value

attr_parsing_invalid_since =
    'since' must be a Rust version number, such as "1.31.0"

attr_parsing_missing_feature =
    missing 'feature'

attr_parsing_missing_issue =
    missing 'issue'

attr_parsing_missing_note =
    missing 'note'

attr_parsing_missing_since =
    missing 'since'

attr_parsing_multiple_stability_levels =
    multiple stability levels

attr_parsing_naked_functions_incompatible_attribute =
    attribute incompatible with `#[unsafe(naked)]`
    .label = the `{$attr}` attribute is incompatible with `#[unsafe(naked)]`
    .naked_attribute = function marked with `#[unsafe(naked)]` here

attr_parsing_non_ident_feature =
    'feature' is not an identifier

attr_parsing_null_on_export = `export_name` may not contain null characters

attr_parsing_repr_ident =
    meta item in `repr` must be an identifier

attr_parsing_rustc_allowed_unstable_pairing =
    `rustc_allowed_through_unstable_modules` attribute must be paired with a `stable` attribute

attr_parsing_rustc_promotable_pairing =
    `rustc_promotable` attribute must be paired with either a `rustc_const_unstable` or a `rustc_const_stable` attribute

attr_parsing_soft_no_args =
    `soft` should not have any arguments

attr_parsing_stability_outside_std = stability attributes may not be used outside of the standard library

attr_parsing_unknown_meta_item =
    unknown meta item '{$item}'
    .label = expected one of {$expected}

attr_parsing_unknown_version_literal =
    unknown version literal format, assuming it refers to a future version

attr_parsing_unrecognized_repr_hint =
    unrecognized representation hint
    .help = valid reprs are `Rust` (default), `C`, `align`, `packed`, `transparent`, `simd`, `i8`, `u8`, `i16`, `u16`, `i32`, `u32`, `i64`, `u64`, `i128`, `u128`, `isize`, `usize`

attr_parsing_unstable_cfg_target_compact =
    compact `cfg(target(..))` is experimental and subject to change

attr_parsing_unsupported_literal_cfg_boolean =
    literal in `cfg` predicate value must be a boolean
attr_parsing_unsupported_literal_cfg_string =
    literal in `cfg` predicate value must be a string
attr_parsing_unsupported_literal_generic =
    unsupported literal
attr_parsing_unsupported_literal_suggestion =
    consider removing the prefix

attr_parsing_unused_duplicate =
    unused attribute
    .suggestion = remove this attribute
    .note = attribute also specified here
    .warn = {-passes_previously_accepted}

attr_parsing_unused_multiple =
    multiple `{$name}` attributes
    .suggestion = remove this attribute
    .note = attribute also specified here

-attr_parsing_perviously_accepted =
    this was previously accepted by the compiler but is being phased out; it will become a hard error in a future release!


attr_parsing_as_needed_compatibility =
    linking modifier `as-needed` is only compatible with `dylib` and `framework` linking kinds

attr_parsing_async_drop_types_in_dependency =
    found async drop types in dependency `{$extern_crate}`, but async_drop feature is disabled for `{$local_crate}`
    .help = if async drop type will be dropped in a crate without `feature(async_drop)`, sync Drop will be used

attr_parsing_bad_panic_strategy =
    the linked panic runtime `{$runtime}` is not compiled with this crate's panic strategy `{$strategy}`

attr_parsing_binary_output_to_tty =
    option `-o` or `--emit` is used to write binary output type `metadata` to stdout, but stdout is a tty

attr_parsing_bundle_needs_static =
    linking modifier `bundle` is only compatible with `static` linking kind

attr_parsing_cannot_find_crate =
    can't find crate for `{$crate_name}`{$add_info}

attr_parsing_cant_find_crate =
    can't find crate

attr_parsing_compiler_missing_profiler =
    the compiler may have been built without the profiler runtime

attr_parsing_conflicting_alloc_error_handler =
    the `#[alloc_error_handler]` in {$other_crate_name} conflicts with allocation error handler in: {$crate_name}

attr_parsing_conflicting_global_alloc =
    the `#[global_allocator]` in {$other_crate_name} conflicts with global allocator in: {$crate_name}

attr_parsing_consider_adding_std =
    consider adding the standard library to the sysroot with `x build library --target {$locator_triple}`

attr_parsing_consider_building_std =
    consider building the standard library from source with `cargo build -Zbuild-std`

attr_parsing_consider_downloading_target =
    consider downloading the target with `rustup target add {$locator_triple}`

attr_parsing_crate_dep_multiple =
    cannot satisfy dependencies so `{$crate_name}` only shows up once
    .help = having upstream crates all available in one format will likely make this go away

attr_parsing_crate_dep_not_static =
    `{$crate_name}` was unavailable as a static crate, preventing fully static linking

attr_parsing_crate_dep_rustc_driver =
    `feature(rustc_private)` is needed to link to the compiler's `rustc_driver` library

attr_parsing_crate_location_unknown_type =
    extern location for {$crate_name} is of an unknown type: {$path}

attr_parsing_crate_not_compiler_builtins =
    the crate `{$crate_name}` resolved as `compiler_builtins` but is not `#![compiler_builtins]`

attr_parsing_crate_not_panic_runtime =
    the crate `{$crate_name}` is not a panic runtime

attr_parsing_dl_error =
    {$path}{$err}

attr_parsing_empty_link_name =
    link name must not be empty
    .label = empty link name

attr_parsing_empty_renaming_target =
    an empty renaming target was specified for library `{$lib_name}`

attr_parsing_extern_location_not_exist =
    extern location for {$crate_name} does not exist: {$location}

attr_parsing_extern_location_not_file =
    extern location for {$crate_name} is not a file: {$location}

attr_parsing_fail_create_file_encoder =
    failed to create file encoder: {$err}

attr_parsing_fail_write_file =
    failed to write to `{$path}`: {$err}

attr_parsing_failed_copy_to_stdout =
    failed to copy {$filename} to stdout: {$err}

attr_parsing_failed_create_encoded_metadata =
    failed to create encoded metadata from file: {$err}

attr_parsing_failed_create_file =
    failed to create the file {$filename}: {$err}

attr_parsing_failed_create_tempdir =
    couldn't create a temp dir: {$err}

attr_parsing_failed_write_error =
    failed to write {$filename}: {$err}

attr_parsing_found_crate_versions =
    the following crate versions were found:{$found_crates}

attr_parsing_found_staticlib =
    found staticlib `{$crate_name}` instead of rlib or dylib{$add_info}
    .help = please recompile that crate using --crate-type lib

attr_parsing_full_metadata_not_found =
    only metadata stub found for `{$flavor}` dependency `{$crate_name}`
    please provide path to the corresponding .rmeta file with full metadata

attr_parsing_global_alloc_required =
    no global memory allocator found but one is required; link to std or add `#[global_allocator]` to a static item that implements the GlobalAlloc trait

attr_parsing_import_name_type_form =
    import name type must be of the form `import_name_type = "string"`

attr_parsing_import_name_type_raw =
    import name type can only be used with link kind `raw-dylib`

attr_parsing_import_name_type_x86 =
    import name type is only supported on x86

attr_parsing_incompatible_panic_in_drop_strategy =
    the crate `{$crate_name}` is compiled with the panic-in-drop strategy `{$found_strategy}` which is incompatible with this crate's strategy of `{$desired_strategy}`

attr_parsing_incompatible_rustc =
    found crate `{$crate_name}` compiled by an incompatible version of rustc{$add_info}
    .help = please recompile that crate using this compiler ({$rustc_version}) (consider running `cargo clean` first)

attr_parsing_incompatible_target_modifiers =
    mixing `{$flag_name_prefixed}` will cause an ABI mismatch in crate `{$local_crate}`
    .note = `{$flag_name_prefixed}={$local_value}` in this crate is incompatible with `{$flag_name_prefixed}={$extern_value}` in dependency `{$extern_crate}`
    .help = the `{$flag_name_prefixed}` flag modifies the ABI so Rust crates compiled with different values of this flag cannot be used together safely
attr_parsing_incompatible_target_modifiers_help_allow = if you are sure this will not cause problems, you may use `-Cunsafe-allow-abi-mismatch={$flag_name}` to silence this error
attr_parsing_incompatible_target_modifiers_help_fix = set `{$flag_name_prefixed}={$extern_value}` in this crate or `{$flag_name_prefixed}={$local_value}` in `{$extern_crate}`

attr_parsing_incompatible_target_modifiers_help_fix_l_missed = set `{$flag_name_prefixed}={$extern_value}` in this crate or unset `{$flag_name_prefixed}` in `{$extern_crate}`

attr_parsing_incompatible_target_modifiers_help_fix_r_missed = unset `{$flag_name_prefixed}` in this crate or set `{$flag_name_prefixed}={$local_value}` in `{$extern_crate}`

attr_parsing_incompatible_target_modifiers_l_missed =
    mixing `{$flag_name_prefixed}` will cause an ABI mismatch in crate `{$local_crate}`
    .note = unset `{$flag_name_prefixed}` in this crate is incompatible with `{$flag_name_prefixed}={$extern_value}` in dependency `{$extern_crate}`
    .help = the `{$flag_name_prefixed}` flag modifies the ABI so Rust crates compiled with different values of this flag cannot be used together safely
attr_parsing_incompatible_target_modifiers_r_missed =
    mixing `{$flag_name_prefixed}` will cause an ABI mismatch in crate `{$local_crate}`
    .note = `{$flag_name_prefixed}={$local_value}` in this crate is incompatible with unset `{$flag_name_prefixed}` in dependency `{$extern_crate}`
    .help = the `{$flag_name_prefixed}` flag modifies the ABI so Rust crates compiled with different values of this flag cannot be used together safely
attr_parsing_incompatible_wasm_link =
    `wasm_import_module` is incompatible with other arguments in `#[link]` attributes

attr_parsing_install_missing_components =
    maybe you need to install the missing components with: `rustup component add rust-src rustc-dev llvm-tools-preview`

attr_parsing_invalid_link_modifier =
    invalid linking modifier syntax, expected '+' or '-' prefix before one of: bundle, verbatim, whole-archive, as-needed

attr_parsing_invalid_meta_files =
    found invalid metadata files for crate `{$crate_name}`{$add_info}

attr_parsing_lib_filename_form =
    file name should be lib*.rlib or {$dll_prefix}*{$dll_suffix}

attr_parsing_lib_framework_apple =
    library kind `framework` is only supported on Apple targets

attr_parsing_lib_required =
    crate `{$crate_name}` required to be available in {$kind} format, but was not found in this form

attr_parsing_link_arg_unstable =
    link kind `link-arg` is unstable

attr_parsing_link_cfg_form =
    link cfg must be of the form `cfg(/* predicate */)`

attr_parsing_link_cfg_single_predicate =
    link cfg must have a single predicate argument

attr_parsing_link_cfg_unstable =
    link cfg is unstable

attr_parsing_link_framework_apple =
    link kind `framework` is only supported on Apple targets

attr_parsing_link_kind_form =
    link kind must be of the form `kind = "string"`

attr_parsing_link_modifiers_form =
    link modifiers must be of the form `modifiers = "string"`

attr_parsing_link_name_form =
    link name must be of the form `name = "string"`

attr_parsing_link_ordinal_raw_dylib =
    `#[link_ordinal]` is only supported if link kind is `raw-dylib`

attr_parsing_link_requires_name =
    `#[link]` attribute requires a `name = "string"` argument
    .label = missing `name` argument

attr_parsing_missing_native_library =
    could not find native static library `{$libname}`, perhaps an -L flag is missing?

attr_parsing_multiple_candidates =
    multiple candidates for `{$flavor}` dependency `{$crate_name}` found

attr_parsing_multiple_cfgs =
    multiple `cfg` arguments in a single `#[link]` attribute

attr_parsing_multiple_import_name_type =
    multiple `import_name_type` arguments in a single `#[link]` attribute

attr_parsing_multiple_kinds_in_link =
    multiple `kind` arguments in a single `#[link]` attribute

attr_parsing_multiple_link_modifiers =
    multiple `modifiers` arguments in a single `#[link]` attribute

attr_parsing_multiple_modifiers =
    multiple `{$modifier}` modifiers in a single `modifiers` argument

attr_parsing_multiple_names_in_link =
    multiple `name` arguments in a single `#[link]` attribute

attr_parsing_multiple_renamings =
    multiple renamings were specified for library `{$lib_name}`

attr_parsing_multiple_wasm_import =
    multiple `wasm_import_module` arguments in a single `#[link]` attribute

attr_parsing_newer_crate_version =
    found possibly newer version of crate `{$crate_name}`{$add_info}
    .note = perhaps that crate needs to be recompiled?

attr_parsing_no_crate_with_triple =
    couldn't find crate `{$crate_name}` with expected target triple {$locator_triple}{$add_info}

attr_parsing_no_link_mod_override =
    overriding linking modifiers from command line is not supported

attr_parsing_no_multiple_alloc_error_handler =
    cannot define multiple allocation error handlers
    .label = cannot define a new allocation error handler

attr_parsing_no_multiple_global_alloc =
    cannot define multiple global allocators
    .label = cannot define a new global allocator

attr_parsing_no_panic_strategy =
    the crate `{$crate_name}` does not have the panic strategy `{$strategy}`

attr_parsing_no_transitive_needs_dep =
    the crate `{$crate_name}` cannot depend on a crate that needs {$needs_crate_name}, but it depends on `{$deps_crate_name}`

attr_parsing_non_ascii_name =
    cannot load a crate with a non-ascii name `{$crate_name}`

attr_parsing_not_profiler_runtime =
    the crate `{$crate_name}` is not a profiler runtime

attr_parsing_only_provide_library_name = only provide the library name `{$suggested_name}`, not the full filename

attr_parsing_prev_alloc_error_handler =
    previous allocation error handler defined here

attr_parsing_prev_global_alloc =
    previous global allocator defined here

attr_parsing_raw_dylib_elf_unstable =
    link kind `raw-dylib` is unstable on ELF platforms

attr_parsing_raw_dylib_no_nul =
    link name must not contain NUL characters if link kind is `raw-dylib`

attr_parsing_raw_dylib_only_windows =
    link kind `raw-dylib` is only supported on Windows targets

attr_parsing_raw_dylib_unsupported_abi =
    ABI not supported by `#[link(kind = "raw-dylib")]` on this architecture

attr_parsing_renaming_no_link =
    renaming of the library `{$lib_name}` was specified, however this crate contains no `#[link(...)]` attributes referencing this library

attr_parsing_required_panic_strategy =
    the crate `{$crate_name}` requires panic strategy `{$found_strategy}` which is incompatible with this crate's strategy of `{$desired_strategy}`

attr_parsing_rlib_required =
    crate `{$crate_name}` required to be available in rlib format, but was not found in this form

attr_parsing_rustc_lib_required =
    crate `{$crate_name}` required to be available in {$kind} format, but was not found in this form
    .note = only .rmeta files are distributed for `rustc_private` crates other than `rustc_driver`
    .help = try adding `extern crate rustc_driver;` at the top level of this crate

attr_parsing_stable_crate_id_collision =
    found crates (`{$crate_name0}` and `{$crate_name1}`) with colliding StableCrateId values

attr_parsing_std_required =
    `std` is required by `{$current_crate}` because it does not declare `#![no_std]`

attr_parsing_symbol_conflicts_current =
    the current crate is indistinguishable from one of its dependencies: it has the same crate-name `{$crate_name}` and was compiled with the same `-C metadata` arguments, so this will result in symbol conflicts between the two

attr_parsing_target_no_std_support =
    the `{$locator_triple}` target may not support the standard library

attr_parsing_target_not_installed =
    the `{$locator_triple}` target may not be installed

attr_parsing_two_panic_runtimes =
    cannot link together two panic runtimes: {$prev_name} and {$cur_name}

attr_parsing_unexpected_link_arg =
    unexpected `#[link]` argument, expected one of: name, kind, modifiers, cfg, wasm_import_module, import_name_type

attr_parsing_unknown_import_name_type =
    unknown import name type `{$import_name_type}`, expected one of: decorated, noprefix, undecorated

attr_parsing_unknown_link_kind =
    unknown link kind `{$kind}`, expected one of: static, dylib, framework, raw-dylib, link-arg
    .label = unknown link kind

attr_parsing_unknown_link_modifier =
    unknown linking modifier `{$modifier}`, expected one of: bundle, verbatim, whole-archive, as-needed

attr_parsing_unknown_target_modifier_unsafe_allowed = unknown target modifier `{$flag_name}`, requested by `-Cunsafe-allow-abi-mismatch={$flag_name}`

attr_parsing_wasm_c_abi =
    older versions of the `wasm-bindgen` crate are incompatible with current versions of Rust; please update to `wasm-bindgen` v0.2.88

attr_parsing_wasm_import_form =
    wasm import module must be of the form `wasm_import_module = "string"`

attr_parsing_whole_archive_needs_static =
    linking modifier `whole-archive` is only compatible with `static` linking kind
