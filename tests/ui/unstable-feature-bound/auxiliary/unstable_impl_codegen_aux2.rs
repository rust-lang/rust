//@ aux-build:unstable_impl_codegen_aux1.rs
#![feature(foo)]

extern crate unstable_impl_codegen_aux1 as aux;
use aux::Trait;

/// Upstream crate for unstable impl codegen test
/// that depends on aux crate in
/// unstable_impl_codegen_aux1.rs

pub fn foo<T>(a: T) {
    a.method();
}
