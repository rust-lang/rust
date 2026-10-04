//@ known-bug: #149809
#![feature(gca_min_const_items)]
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
