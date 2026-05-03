//@ known-bug: unknown
//@ run-pass

#![feature(view_types, view_type_macro)]
#![allow(unused)]

use std::view::view_type;

struct Foo {
    bar: usize,
}

struct Pair(usize);

fn f(_foo: &mut view_type!(Foo.{ bar, bar }), _pair: &mut view_type!(Pair.{ 0, 0 })) {}

impl Foo {
    fn f(self: &mut view_type!(Self.{ bar, bar })) {}
}

impl Pair {
    fn f(self: &mut view_type!(Self.{ 0, 0 })) {}
}

fn main() {}
