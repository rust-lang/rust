// aux-build:intra-doc-broken.rs
// check-pass

#![deny(broken_intra_doc_links)]

extern crate intra_doc_broken;

pub use intra_doc_broken::foo;
