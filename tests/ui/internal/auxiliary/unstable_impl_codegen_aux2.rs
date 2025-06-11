//@ aux-build:unstable_impl_codegen_aux1.rs

#![allow(internal_features)]
#![feature(staged_api)]
#![feature(impl_stability)]
#![stable(feature = "a", since = "1.1.1" )]
#[feature(foo)]

extern crate unstable_impl_codegen_aux1 as aux;

/// Upstream crate for unstable impl codegen test 
/// that depends on aux crate in 
/// unstable_impl_codegen_aux1.rs

fn foo<T>(a:T) {
    a.method();
}

fn main() {
}
