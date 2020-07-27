// aux-build:additional_doc.rs
// build-aux-docs
#![deny(intra_doc_link_resolution_failure)]

extern crate my_rand;

// @has 'additional_doc/trait.Rng.html' '//a[@href="../additional_doc/trait.Rng.html"]' 'Rng'
// @has 'additional_doc/trait.Rng.html' '//a[@href="../my_rand/trait.RngCore.html"]' 'RngCore'
/// This is an [`Rng`].
pub use my_rand::Rng;
