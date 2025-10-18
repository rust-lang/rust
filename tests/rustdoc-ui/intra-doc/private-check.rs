// This test ensures that private items intra-doc links are checked whereas `doc(hidden)`
// items are ignored.

//@ compile-flags: -Zunstable-options --document-private-items

#![deny(rustdoc::broken_intra_doc_links)]

/// [not::exist]
#[doc(hidden)]
pub struct X;

/// [not::exist]
//~^ ERROR unresolved link to `not::exist`
struct Y;
