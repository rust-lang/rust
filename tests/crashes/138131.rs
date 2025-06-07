//@ known-bug: #138131

#![feature(min_generic_const_args, generic_const_items)]

const BAR<'a>: usize = 10;

struct Foo<'a> {
    x: &'a (),
}

impl<'a> Foo<'a> {
    fn foo(_: [u8; BAR]) {}
}

fn main() {}
