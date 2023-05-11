// aux-build:intra-links-external-traits.rs
// ignore-cross-compile

#![crate_name = "outer"]
#![deny(rustdoc::broken_intra_doc_links)]

// using a trait that has intra-doc links on it from another crate (whether re-exporting or just
// implementing it) used to give spurious resolution failure warnings

extern crate intra_links_external_traits;

pub use intra_links_external_traits::ThisTrait;
