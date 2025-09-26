//@ revisions: old next
//@ [next] compile-flags: -Znext-solver
#![feature(field_projections)]
#![allow(incomplete_features)]
use std::field::field_of;

struct Foo {
    field: u32,
}
struct Bar {
    bar_field: u32,
}
unsafe impl Send for field_of!(Bar, bar_field) {}
//~^ ERROR: impls of auto traits for field representing types not supported

fn is_send<T: Send>() {}

fn main() {
    is_send::<field_of!(Bar, bar_field)>();
    is_send::<field_of!(Foo, field)>();
    //~^ ERROR: `field_of!(Foo, field)` cannot be sent between threads safely [E0277]
}
