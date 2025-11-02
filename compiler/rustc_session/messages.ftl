session_apple_deployment_target_invalid =
    failed to parse deployment target specified in {$env_var}: {$error}

session_apple_deployment_target_too_low =
    deployment target in {$env_var} was set to {$version}, but the minimum supported by `rustc` is {$os_min}

session_binary_float_literal_not_supported = binary float literal is not supported
session_branch_protection_requires_aarch64 = `-Zbranch-protection` is only supported on aarch64

session_cannot_enable_crt_static_linux = sanitizer is incompatible with statically linked libc, disable it using `-C target-feature=-crt-static`

session_cannot_mix_and_match_sanitizers = `-Zsanitizer={$first}` is incompatible with `-Zsanitizer={$second}`

session_cli_feature_diagnostic_help =
    add `-Zcrate-attr="feature({$feature})"` to the command-line options to enable

session_crate_name_empty = crate name must not be empty

session_embed_source_insufficient_dwarf_version = `-Zembed-source=y` requires at least `-Z dwarf-version=5` but DWARF version is {$dwarf_version}

session_embed_source_requires_debug_info = `-Zembed-source=y` requires debug information to be enabled

session_expr_parentheses_needed = parentheses are required to parse this as an expression

session_failed_to_create_profiler = failed to create profiler: {$err}

session_feature_diagnostic_for_issue =
    see issue #{$n} <https://github.com/rust-lang/rust/issues/{$n}> for more information

session_feature_diagnostic_help =
    add `#![feature({$feature})]` to the crate attributes to enable

session_feature_diagnostic_suggestion =
    add `#![feature({$feature})]` to the crate attributes to enable

session_feature_suggest_upgrade_compiler =
    this compiler was built on {$date}; consider upgrading it if it is out of date

session_file_is_not_writeable = output file {$file} is not writeable -- check its permissions

session_file_write_fail = failed to write `{$path}` due to error `{$err}`

session_function_return_requires_x86_or_x86_64 = `-Zfunction-return` (except `keep`) is only supported on x86 and x86_64

session_function_return_thunk_extern_requires_non_large_code_model = `-Zfunction-return=thunk-extern` is only supported on non-large code models

session_hexadecimal_float_literal_not_supported = hexadecimal float literal is not supported

session_incompatible_linker_flavor = linker flavor `{$flavor}` is incompatible with the current target
    .note = compatible flavors are: {$compatible_list}

session_indirect_branch_cs_prefix_requires_x86_or_x86_64 = `-Zindirect-branch-cs-prefix` is only supported on x86 and x86_64

session_instrumentation_not_supported = {$us} instrumentation is not supported for this target

session_int_literal_too_large = integer literal is too large
    .note = value exceeds limit of `{$limit}`

session_invalid_character_in_crate_name = invalid character {$character} in crate name: `{$crate_name}`
    .help = you can either pass `--crate-name` on the command line or add `#![crate_name = "…"]` to set the crate name

session_invalid_float_literal_suffix = invalid suffix `{$suffix}` for float literal
    .label = invalid suffix `{$suffix}`
    .help = valid suffixes are `f32` and `f64`

session_invalid_float_literal_width = invalid width `{$width}` for float literal
    .help = valid widths are 32 and 64

session_invalid_int_literal_width = invalid width `{$width}` for integer literal
    .help = valid widths are 8, 16, 32, 64 and 128

session_invalid_literal_suffix = suffixes on {$kind} literals are invalid
    .label = invalid suffix `{$suffix}`

session_invalid_num_literal_base_prefix = invalid base prefix for number literal
    .note = base prefixes (`0xff`, `0b1010`, `0o755`) are lowercase
    .suggestion = try making the prefix lowercase

session_invalid_num_literal_suffix = invalid suffix `{$suffix}` for number literal
    .label = invalid suffix `{$suffix}`
    .help = the suffix must be one of the numeric types (`u32`, `isize`, `f32`, etc.)

session_linker_plugin_lto_windows_not_supported = linker plugin based LTO is not supported together with `-C prefer-dynamic` when targeting Windows-like targets

session_not_circumvent_feature = `-Zunleash-the-miri-inside-of-you` may not be used to circumvent feature gates, except when testing error paths in the CTFE engine

session_not_supported = not supported

session_octal_float_literal_not_supported = octal float literal is not supported

session_profile_sample_use_file_does_not_exist = file `{$path}` passed to `-C profile-sample-use` does not exist

session_profile_use_file_does_not_exist = file `{$path}` passed to `-C profile-use` does not exist

session_sanitizer_cfi_canonical_jump_tables_requires_cfi = `-Zsanitizer-cfi-canonical-jump-tables` requires `-Zsanitizer=cfi`

session_sanitizer_cfi_generalize_pointers_requires_cfi = `-Zsanitizer-cfi-generalize-pointers` requires `-Zsanitizer=cfi` or `-Zsanitizer=kcfi`

session_sanitizer_cfi_normalize_integers_requires_cfi = `-Zsanitizer-cfi-normalize-integers` requires `-Zsanitizer=cfi` or `-Zsanitizer=kcfi`

session_sanitizer_cfi_requires_lto = `-Zsanitizer=cfi` requires `-Clto` or `-Clinker-plugin-lto`

session_sanitizer_cfi_requires_single_codegen_unit = `-Zsanitizer=cfi` with `-Clto` requires `-Ccodegen-units=1`

session_sanitizer_kcfi_arity_requires_kcfi = `-Zsanitizer-kcfi-arity` requires `-Zsanitizer=kcfi`

session_sanitizer_kcfi_requires_panic_abort = `-Z sanitizer=kcfi` requires `-C panic=abort`

session_sanitizer_not_supported = {$us} sanitizer is not supported for this target

session_sanitizers_not_supported = {$us} sanitizers are not supported for this target

session_skipping_const_checks = skipping const checks

session_soft_float_deprecated =
    `-Csoft-float` is unsound and deprecated; use a corresponding *eabi target instead
    .note = it will be removed or ignored in a future version of Rust
session_soft_float_deprecated_issue = see issue #129893 <https://github.com/rust-lang/rust/issues/129893> for more information

session_soft_float_ignored =
    `-Csoft-float` is ignored on this target; it only has an effect on *eabihf targets
    .note = this may become a hard error in a future version of Rust

session_split_debuginfo_unstable_platform = `-Csplit-debuginfo={$debuginfo}` is unstable on this platform

session_split_lto_unit_requires_lto = `-Zsplit-lto-unit` requires `-Clto`, `-Clto=thin`, or `-Clinker-plugin-lto`

session_target_requires_unwind_tables = target requires unwind tables, they cannot be disabled with `-C force-unwind-tables=no`

session_target_small_data_threshold_not_supported = `-Z small-data-threshold` is not supported for target {$target_triple} and will be ignored

session_target_stack_protector_not_supported = `-Z stack-protector={$stack_protector}` is not supported for target {$target_triple} and will be ignored

session_unexpected_builtin_cfg = unexpected `--cfg {$cfg}` flag
    .controlled_by = config `{$cfg_name}` is only supposed to be controlled by `{$controlled_by}`
    .incoherent = manually setting a built-in cfg can and does create incoherent behaviors

session_unleashed_feature_help_named = skipping check for `{$gate}` feature
session_unleashed_feature_help_unnamed = skipping check that does not even have a feature gate

session_unstable_virtual_function_elimination = `-Zvirtual-function-elimination` requires `-Clto`

session_unsupported_crate_type_for_target =
    dropping unsupported crate type `{$crate_type}` for target `{$target_triple}`

session_unsupported_dwarf_version = requested DWARF version {$dwarf_version} is not supported
session_unsupported_dwarf_version_help = supported DWARF versions are 2, 3, 4 and 5

session_unsupported_reg_struct_return_arch = `-Zreg-struct-return` is only supported on x86
session_unsupported_regparm = `-Zregparm={$regparm}` is unsupported (valid values 0-3)
session_unsupported_regparm_arch = `-Zregparm=N` is only supported on x86
