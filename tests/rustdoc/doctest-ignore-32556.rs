#![crate_name="issue_32556"]

/// Blah blah blah
/// ```ignore (testing rustdoc's handling of ignore)
/// bad_assert!();
/// ```
pub fn foo() {}
