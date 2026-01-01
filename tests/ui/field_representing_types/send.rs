//@ revisions: old next
//@ [next] compile-flags: -Znext-solver
//@ run-pass
#![feature(field_projections)]
#![allow(incomplete_features, dead_code)]
use std::field::field_of;

struct Foo {
    field: u32,
}
struct Bar {
    bar_field: u32,
}

fn is_send<T: Send>() {}

fn main() {
    is_send::<field_of!(Bar, bar_field)>();
    is_send::<field_of!(Foo, field)>();
}
