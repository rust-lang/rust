// compile-flags:--test

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
