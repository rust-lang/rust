#![feature(view_types, view_type_macro)]
#![allow(unused)]

use std::view::view_type;

struct Foo {
    bar: usize,
}

struct Pair(usize);

fn f(
    _foo: &mut view_type!(Foo.{ bar, bar }),
    //~^ ERROR field `bar` is already part of the view
    _pair: &mut view_type!(Pair.{ 0, 0 }),
    //~^ ERROR field `0` is already part of the view
) {
}

impl Foo {
    fn f(self: &mut view_type!(Self.{ bar, bar })) {}
    //~^ ERROR field `bar` is already part of the view
}

impl Pair {
    fn f(self: &mut view_type!(Self.{ 0, 0 })) {}
    //~^ ERROR field `0` is already part of the view
}

fn main() {}
