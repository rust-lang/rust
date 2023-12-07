session_branch_protection_requires_aarch64 = `-Zbranch-protection` is only supported on aarch64

session_cannot_enable_crt_static_linux = sanitizer is incompatible with statically linked libc, disable it using `-C target-feature=-crt-static`

session_cannot_mix_and_match_sanitizers = `-Zsanitizer={$first}` is incompatible with `-Zsanitizer={$second}`

session_cli_feature_diagnostic_help =
    add `-Zcrate-attr="feature({$feature})"` to the command-line options to enable

session_crate_name_does_not_match = `--crate-name` and `#[crate_name]` are required to match, but `{$s}` != `{$name}`

session_crate_name_empty = crate name must not be empty

session_crate_name_invalid = crate names cannot start with a `-`, but `{$s}` has a leading hyphen

session_expr_parentheses_needed = parentheses are required to parse this as an expression

session_failed_to_create_profiler = failed to create profiler: {$err}

session_feature_diagnostic_for_issue =
    see issue #{$n} <https://github.com/rust-lang/rust/issues/{$n}> for more information

session_feature_diagnostic_help =
    add `#![feature({$feature})]` to the crate attributes to enable

session_file_is_not_writeable = output file {$file} is not writeable -- check its permissions

session_file_write_fail = failed to write `{$path}` due to error `{$err}`

session_function_return_requires_x86_or_x86_64 = `-Zfunction-return` (except `keep`) is only supported on x86 and x86_64

session_function_return_thunk_extern_requires_non_large_code_model = `-Zfunction-return=thunk-extern` is only supported on non-large code models

session_incompatible_linker_flavor = linker flavor `{$flavor}` is incompatible with the current target
    .note = compatible flavors are: {$compatible_list}

session_instrumentation_not_supported = {$us} instrumentation is not supported for this target

session_invalid_character_in_create_name = invalid character `{$character}` in crate name: `{$crate_name}`
session_invalid_character_in_create_name_help = you can either pass `--crate-name` on the command line or add `#![crate_name="…"]` to set the crate name

session_linker_plugin_lto_windows_not_supported = linker plugin based LTO is not supported together with `-C prefer-dynamic` when targeting Windows-like targets

session_not_circumvent_feature = `-Zunleash-the-miri-inside-of-you` may not be used to circumvent feature gates, except when testing error paths in the CTFE engine

session_optimization_fuel_exhausted = optimization-fuel-exhausted: {$msg}

session_profile_sample_use_file_does_not_exist = file `{$path}` passed to `-C profile-sample-use` does not exist.

session_profile_use_file_does_not_exist = file `{$path}` passed to `-C profile-use` does not exist.

session_sanitizer_cfi_canonical_jump_tables_requires_cfi = `-Zsanitizer-cfi-canonical-jump-tables` requires `-Zsanitizer=cfi`

session_sanitizer_cfi_generalize_pointers_requires_cfi = `-Zsanitizer-cfi-generalize-pointers` requires `-Zsanitizer=cfi` or `-Zsanitizer=kcfi`

session_sanitizer_cfi_normalize_integers_requires_cfi = `-Zsanitizer-cfi-normalize-integers` requires `-Zsanitizer=cfi` or `-Zsanitizer=kcfi`

session_sanitizer_cfi_requires_lto = `-Zsanitizer=cfi` requires `-Clto` or `-Clinker-plugin-lto`

session_sanitizer_cfi_requires_single_codegen_unit = `-Zsanitizer=cfi` with `-Clto` requires `-Ccodegen-units=1`

session_sanitizer_not_supported = {$us} sanitizer is not supported for this target

session_sanitizers_not_supported = {$us} sanitizers are not supported for this target

session_skipping_const_checks = skipping const checks
session_split_debuginfo_unstable_platform = `-Csplit-debuginfo={$debuginfo}` is unstable on this platform

session_split_lto_unit_requires_lto = `-Zsplit-lto-unit` requires `-Clto`, `-Clto=thin`, or `-Clinker-plugin-lto`

session_target_requires_unwind_tables = target requires unwind tables, they cannot be disabled with `-C force-unwind-tables=no`

session_target_stack_protector_not_supported = `-Z stack-protector={$stack_protector}` is not supported for target {$target_triple} and will be ignored

session_unleashed_feature_help_named = skipping check for `{$gate}` feature
session_unleashed_feature_help_unnamed = skipping check that does not even have a feature gate

session_unstable_virtual_function_elimination = `-Zvirtual-function-elimination` requires `-Clto`

session_unsupported_dwarf_version = requested DWARF version {$dwarf_version} is greater than 5
