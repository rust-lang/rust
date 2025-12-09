//! regression test for #148949, #149809
#![expect(incomplete_features)]
#![feature(min_generic_const_args)]
#![feature(inherent_associated_types)]
struct Foo<'a> {
    x: &'a (),
}

impl<'a> Foo<'a> {
    fn bar(_: [u8; Foo::X]) {}
    //~^ ERROR: associated constant `X` not found for `Foo<'_>` in the current scope
    // #[type_const]
    // const Y: usize = 10;
    // fn foo(_: [u8; Foo::Y]) {}
    // associated constant `Y` not found for `Foo<'_>` in the current scope
}

fn main() {}
