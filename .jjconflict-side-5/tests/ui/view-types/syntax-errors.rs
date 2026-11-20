#![feature(view_types, view_type_macro)]
#![allow(unused)]

use std::view::view_type;

struct Foo {
    bar: usize,
    baz: usize,
}

impl Foo {
    fn not_a_field(self: &mut view_type!(Self.{ _ }), _: &mut view_type!(Foo.{ _ })) {}
    //~^ ERROR expected identifier, found reserved identifier
    //~| ERROR expected identifier, found reserved identifier
    //~| ERROR no field `_` on type `Foo`
    //~| ERROR no field `_` on type `Foo`

    fn keyword(self: &mut view_type!(Foo.{ where }), _: &mut view_type!(Foo.{ for })) {}
    //~^ ERROR expected identifier
    //~| ERROR expected identifier
    //~| ERROR no field `r#where` on type `Foo`
    //~| ERROR no field `r#for` on type `Foo`

    fn no_comma(self: &mut view_type!(Foo.{ bar baz }), _: &mut view_type!(Foo.{ bar baz })) {}
    //~^ ERROR expected one of
    //~| ERROR expected one of
}

fn main() {}
