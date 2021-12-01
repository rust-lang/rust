// aux-build:additional_doc.rs
// build-aux-docs
#![deny(rustdoc::broken_intra_doc_links)]

extern crate my_rand;

// @has 'additional_doc/trait.Rng.html' '//a[@href="trait.Rng.html"]' 'Rng'
// @has 'additional_doc/trait.Rng.html' '//a[@href="../my_rand/trait.RngCore.html"]' 'RngCore'
/// This is an [`Rng`].
pub use my_rand::Rng;
