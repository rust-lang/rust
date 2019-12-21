#![crate_name="lint_output_format"]
#![crate_type = "lib"]
#![feature(staged_api)]
#![unstable(feature = "unstable_test_feature", issue = "none")]

#[stable(feature = "stable_test_feature", since = "1.0.0")]
#[rustc_deprecated(since = "1.0.0", reason = "text")]
pub fn foo() -> usize {
    20
}

#[unstable(feature = "unstable_test_feature", issue = "none")]
pub fn bar() -> usize {
    40
}

#[unstable(feature = "unstable_test_feature", issue = "none")]
pub fn baz() -> usize {
    30
}
