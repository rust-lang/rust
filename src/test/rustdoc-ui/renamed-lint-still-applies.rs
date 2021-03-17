// compile-args: --crate-type lib
#![deny(broken_intra_doc_links)]
// FIXME: the old names for rustdoc lints should warn by default once `rustdoc::` makes it to the
// stable channel.
//! [x]
//~^ ERROR unresolved link
