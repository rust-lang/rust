//@ edition: 2024
//@ check-fail

#![crate_type = "rlib"]
#![feature(rustc_private)]

extern crate rustc_type_ir;
extern crate rustc_type_ir_macros;

use rustc_type_ir_macros::GenericTypeVisitable;

#[derive(GenericTypeVisitable)]
struct MissingBound<T> {
    // This should fail, as `T: GenericTypeVisitable<__V>` wasn't specified
    #[generic_type_visitable(bounds())]
    //~^ ERROR: the trait bound `T: GenericTypeVisitable<__V>` is not satisfied
    partially_rec: (Vec<Self>, T),
    other: u32,
}
