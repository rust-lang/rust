#![feature(view_types, view_type_macro)]
//~^ ERROR unknown feature `view_type_macro`
#![allow(unused)]

use std::view::view_type;
//~^ ERROR unresolved import

struct Foo {
    bar: usize,
    baz: usize,
}

impl Foo {
    fn not_a_field(self: &mut view_type!(Foo.{ _ }), _: &mut view_type!(Foo.{ _ })) {}
    //~^ ERROR invalid `self` parameter type

    fn keyword(self: &mut view_type!(Foo.{ where }), _: &mut view_type!(Foo.{ for })) {}
    //~^ ERROR invalid `self` parameter type

    fn no_comma(self: &mut view_type!(Foo.{ bar baz }), _: &mut view_type!(Foo.{ bar baz })) {}
    //~^ ERROR invalid `self` parameter type
}

fn main() {}
