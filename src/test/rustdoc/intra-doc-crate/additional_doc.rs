// aux-build:additional_doc.rs
// build-aux-docs
#![deny(intra_doc_resolution_failure)]

extern crate rand;

// @has 'additional_doc/trait.Rng.html' '//a[@href="../additional_doc/trait.Rng.html"]' 'Rng'
// @has 'additional_doc/trait.Rng.html' '//a[@href="../rand/trait.RngCore.html"]' 'RngCore'
/// This is an [`Rng`].
pub use rand::Rng;
