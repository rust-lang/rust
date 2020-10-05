// revisions: full min
#![cfg_attr(full, feature(const_generics))]
#![cfg_attr(full, allow(incomplete_features))]
#![cfg_attr(min, feature(min_const_generics))]

trait Foo {}

impl<const N: usize> Foo for [(); N]
    where
        Self:FooImpl<{N==0}>
//[full]~^ERROR constant expression depends on a generic parameter
//[min]~^^ERROR generic parameters must not be used inside of non-trivial constant values
{}

trait FooImpl<const IS_ZERO: bool>{}

impl FooImpl<true> for [(); 0] {}

impl<const N:usize> FooImpl<false> for [();N] {}

fn foo(_: impl Foo) {}

fn main() {
    foo([]);
    foo([()]);
}
