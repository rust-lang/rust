//@ compile-flags:--test
//@ normalize-stdout: "finished in \d+\.\d+s" -> "finished in $$TIME"
//@ check-pass

// https://github.com/rust-lang/rust/issues/23106
#![crate_name="issue_23106"]

/// ```
/// #
/// ```
pub fn main() {
}
