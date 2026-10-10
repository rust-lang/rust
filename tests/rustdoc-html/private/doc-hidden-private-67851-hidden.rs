//@ compile-flags: -Zunstable-options --document-hidden-items
// https://github.com/rust-lang/rust/issues/67851
#![crate_name="foo"]

//@ has foo/type.Hidden.html
#[doc(hidden)]
pub struct Hidden;

//@ !has foo/type.Private.html
struct Private;
