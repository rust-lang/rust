// https://github.com/rust-lang/rust/issues/32556
#![crate_name="issue_32556"]

/// Blah blah blah
/// ```ignore (testing rustdoc's handling of ignore)
/// bad_assert!();
/// ```
pub fn foo() {}
