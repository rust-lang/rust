//@ run-pass

#![feature(view_types, view_type_macro)]
#![allow(unused)]

use std::view::view_type;

struct Pair(usize, u32);

impl Pair {
    fn foo(self: &mut view_type!(Pair.{ 0, 1 })) {}
    fn bar(_pair: &mut view_type!(Pair.{ 0, 1 })) {}
}

fn main() {}
