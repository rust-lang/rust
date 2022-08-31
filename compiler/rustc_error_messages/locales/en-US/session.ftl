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
