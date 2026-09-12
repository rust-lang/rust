//@ compile-flags:--test
//@ normalize-stdout: "finished in \d+\.\d+s" -> "finished in $$TIME"
//@ check-pass

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
