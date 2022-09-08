session_incorrect_cgu_reuse_type =
    CGU-reuse for `{$cgu_user_name}` is `{$actual_reuse}` but should be {$at_least ->
    [one] {"at least "}
    *[other] {""}
    }`{$expected_reuse}`

session_cgu_not_recorded =
    CGU-reuse for `{$cgu_user_name}` is (mangled: `{$cgu_name}`) was not recorded`

session_feature_gate_error = {$explain}

session_feature_diagnostic_for_issue =
    see issue #{$n} <https://github.com/rust-lang/rust/issues/{$n}> for more information

session_feature_diagnostic_help =
    add `#![feature({$feature})]` to the crate attributes to enable

session_not_circumvent_feature = `-Zunleash-the-miri-inside-of-you` may not be used to circumvent feature gates, except when testing error paths in the CTFE engine

session_profile_use_file_does_not_exist = file `{$path}` passed to `-C profile-use` does not exist.

session_linker_plugin_lto_windows_not_supported = linker plugin based LTO is not supported together with `-C prefer-dynamic` when targeting Windows-like targets

session_profile_sample_use_file_does_not_exist = file `{$path}` passed to `-C profile-sample-use` does not exist.

session_target_requires_unwind_tables = target requires unwind tables, they cannot be disabled with `-C force-unwind-tables=no`

session_sanitizer_not_supported = {$us} sanitizer is not supported for this target

session_sanitizers_not_supported = {$us} sanitizers are not supported for this target

session_cannot_mix_and_match_sanitizers = `-Zsanitizer={$first}` is incompatible with `-Zsanitizer={$second}`

session_cannot_enable_crt_static_linux = sanitizer is incompatible with statically linked libc, disable it using `-C target-feature=-crt-static`

session_sanitizer_cfi_enabled = `-Zsanitizer=cfi` requires `-Clto`

session_unstable_virtual_function_elimination = `-Zvirtual-function-elimination` requires `-Clto`

session_unsupported_dwarf_version = requested DWARF version {$dwarf_version} is greater than 5

session_target_invalid_address_space = invalid address space `{$addr_space}` for `{$cause}` in "data-layout": {$err}

session_target_invalid_bits = invalid {$kind} `{$bit}` for `{$cause}` in "data-layout": {$err}

session_target_missing_alignment = missing alignment for `{$cause}` in "data-layout"

session_target_invalid_alignment = invalid alignment for `{$cause}` in "data-layout": {$err}

session_target_inconsistent_architecture = inconsistent target specification: "data-layout" claims architecture is {$dl}-endian, while "target-endian" is `{$target}`

session_target_inconsistent_pointer_width = inconsistent target specification: "data-layout" claims pointers are {$pointer_size}-bit, while "target-pointer-width" is `{$target}`

session_target_invalid_bits_size = {$err}

session_target_stack_protector_not_supported = `-Z stack-protector={$stack_protector}` is not supported for target {$target_triple} and will be ignored

session_split_debuginfo_unstable_platform = `-Csplit-debuginfo={$debuginfo}` is unstable on this platform
