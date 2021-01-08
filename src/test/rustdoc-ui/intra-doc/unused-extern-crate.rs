// compile-flags: --extern zip=whatever.rlib
#![deny(broken_intra_doc_links)]
/// See [zip] crate.
//~^ ERROR unresolved
pub struct ArrayZip;
