//@ known-bug: #150049
#![feature(gca_min_const_items)]
#![feature(inherent_associated_types)]

use std::gca;

struct Foo<'a> {
    x: &'a (),
}

impl<'a> Foo<'a> {
    fn foo(_: [u8; gca!(Foo::X)]) {
        std::mem::transmute([4])
    }
}

fn main() {}
