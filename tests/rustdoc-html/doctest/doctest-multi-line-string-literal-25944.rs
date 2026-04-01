//@ compile-flags:--test

// https://github.com/rust-lang/rust/issues/25944
#![crate_name="issue_25944"]

/// ```
/// let a = r#"
/// foo
/// bar"#;
/// let b = "\nfoo\nbar";
/// assert_eq!(a, b);
/// ```
pub fn main() {
}
