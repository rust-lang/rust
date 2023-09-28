// compile-flags: -Z unstable-options --document-hidden-items

#![crate_name="issue_15347"]

// @has issue_15347/fn.foo.html
#[doc(hidden)]
pub fn foo() {}
