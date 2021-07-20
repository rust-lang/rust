// Checks to ensure that we properly detect when a closure constrains an opaque type

// revisions: min_tait full_tait
#![feature(min_type_alias_impl_trait)]
#![cfg_attr(full_tait, feature(type_alias_impl_trait))]
//[full_tait]~^ WARN incomplete

use std::fmt::Debug;

fn main() {
    type Opaque = impl Debug;
    fn _unused() -> Opaque { String::new() }
    let null = || -> Opaque { 0 }; //[min_tait]~ ERROR: concrete type differs from previous defining opaque type use
    //[full_tait]~^ ERROR: concrete type differs from previous defining opaque type use
    //[min_tait]~^^ ERROR: type alias impl trait is not permitted here
    println!("{:?}", null());
}
