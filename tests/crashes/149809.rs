//@ known-bug: #149809
#![feature(min_generic_const_args)]
#![feature(inherent_associated_types)]

use std::gca;

struct Qux<'a> {
    x: &'a (),
}

impl<'a> Qux<'a> {
    const LEN: usize = gca!(4);
    fn foo(_: [u8; gca!(Qux::LEN)]) {}
}

fn main() {}
