//@ compile-flags: -Zunstable-options --document-private-items --document-hidden-items
// https://github.com/rust-lang/rust/issues/67851
#![crate_name="foo"]

//@ has foo/struct.Hidden.html
#[doc(hidden)]
pub struct Hidden;

//@ has foo/struct.Private.html
struct Private;
