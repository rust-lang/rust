//@ compile-flags: -Z unstable-options --document-hidden-items
// https://github.com/rust-lang/rust/issues/15347

#![crate_name="issue_15347"]

//@ has issue_15347/fn.foo.html
#[doc(hidden)]
pub fn foo() {}
