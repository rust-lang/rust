//@ run-pass
#![allow(unused_variables)]
//@ aux-build:trait-object-projection-bounds-with-interning-order.rs

//! Regression test for https://github.com/rust-lang/rust/issues/25467

pub type Issue25467BarT = ();
pub type Issue25467FooT = ();

extern crate trait_object_projection_bounds_with_interning_order as aux;

fn main() {
    let o: aux::Object = None;
}
