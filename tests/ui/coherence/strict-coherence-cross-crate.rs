//! Impl a `#[rustc_strict_coherence]` trait from a dependency, without the feature.
//@ aux-build: strict-coherence-cross-crate-aux.rs
//@ check-pass

extern crate strict_coherence_cross_crate_aux as aux;

struct L;

impl aux::Tr for L {}

fn main() {}
