// This test ensures that `doc(hidden)` items intra-doc links are checked whereas private
// items are ignored.

//@ compile-flags: -Zunstable-options --document-hidden-items

#![deny(rustdoc::broken_intra_doc_links)]

/// [not::exist]
//~^ ERROR unresolved link to `not::exist`
#[doc(hidden)]
pub struct X;

/// [not::exist]
struct Y;
