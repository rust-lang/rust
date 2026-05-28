//@ revisions: old next
//@ [next] compile-flags: -Znext-solver
#![expect(incomplete_features)]
#![feature(field_projections)]

use std::field::field_of;

pub struct Struct<'a> {
    field: &'a (),
}

fn consume<'a>(_: field_of!(Struct<'a>, field), _: field_of!(Struct<'a>, field)) {}

fn assert_invariant<'a, 'b>(x: field_of!(Struct<'a>, field), y: field_of!(Struct<'b>, field)) {
    consume(x, y);
    //~^ ERROR: lifetime may not live long enough
    //~^^ ERROR: lifetime may not live long enough
}

fn main() {}
