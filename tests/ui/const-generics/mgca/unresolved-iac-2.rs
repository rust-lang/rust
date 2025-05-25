#![feature(min_generic_const_args)]
#![feature(inherent_associated_types)]
#![expect(incomplete_features)]

struct Foo<'a> {
    x: &'a (),
}

impl<'a> Foo<'a> {
    fn foo(_: [u8; Foo::X]) {}
    //~^ ERROR: associated constant `X` not found for `Foo<'_>` in the current scope
}

fn main() {}
