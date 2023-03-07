// compile-flags: -Z unstable-options --document-hidden-items

// @has issue_15347/fn.foo.html
#[doc(hidden)]
pub fn foo() {}
