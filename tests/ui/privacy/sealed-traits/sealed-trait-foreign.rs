//@ aux-build:sealed-trait-foreign-aux.rs

extern crate sealed_trait_foreign_aux as aux;

struct S;
impl aux::Sealed for S {} //~ ERROR the trait bound
impl aux::NotSealed for S {} //~ ERROR the trait bound

fn main() {}
