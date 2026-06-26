// https://github.com/rust-lang/rust/issues/32556
//@ compile-flags:--test
//@ normalize-stdout: "finished in \d+\.\d+s" -> "finished in $$TIME"
//@ check-pass

#![crate_name="issue_32556"]

/// Blah blah blah
/// ```ignore (testing rustdoc's handling of ignore)
/// bad_assert!();
/// ```
pub fn foo() {}
