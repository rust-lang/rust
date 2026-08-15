//@ known-bug: #150049
#![feature(min_generic_const_args)]
#![feature(inherent_associated_types)]
struct Foo<'a> {
    x: &'a (),
}

impl<'a> Foo<'a> {
    fn foo(_: [u8; core::direct_const_arg!(Foo::X)]) {
        std::mem::transmute([4])
    }
}

fn main() {}
