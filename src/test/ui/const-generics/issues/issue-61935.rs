// check-pass

#![feature(const_generics)]
//~^ WARN the feature `const_generics` is incomplete

trait Foo {}

impl<const N: usize> Foo for [(); N]
    where
        Self:FooImpl<{N==0}>
{}

trait FooImpl<const IS_ZERO: bool>{}

impl FooImpl<true> for [(); 0] {}

impl<const N:usize> FooImpl<false> for [();N] {}

fn foo(_: impl Foo) {}

fn main() {
    foo([]);
    foo([()]);
}
