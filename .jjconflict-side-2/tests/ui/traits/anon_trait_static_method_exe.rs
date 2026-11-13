//@ run-pass
#![allow(non_camel_case_types)]

//@ aux-build:anon_trait_static_method_lib.rs

extern crate anon_trait_static_method_lib;
use anon_trait_static_method_lib::Foo;

pub fn main() {
    let x = Foo::new();
    println!("{}", x.x);
}
