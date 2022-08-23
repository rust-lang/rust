metadata_rlib_required =
    crate `{$crate_name}` required to be available in rlib format, but was not found in this form

metadata_lib_required =
    crate `{$crate_name}` required to be available in {$kind} format, but was not found in this form

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
