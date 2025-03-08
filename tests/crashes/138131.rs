//@ known-bug: #138131
#![feature(min_generic_const_args)]
#![feature(inherent_associated_types)]
struct Foo<'a> {
    x: &'a (),
}

impl<'a> Foo<'a> {
    fn foo(_: [u8; Foo::X]) {}
}

fn main() {}
