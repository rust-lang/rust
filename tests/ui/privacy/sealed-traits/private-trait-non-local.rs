//@ aux-build: private-trait-non-local-aux.rs

extern crate private_trait_non_local_aux as aux;
use aux::a::b::Sealed; //~ ERROR module `b` is private

fn main() {}
