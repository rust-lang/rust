// revisions: full min
#![cfg_attr(full, feature(const_generics))]
#![cfg_attr(full, allow(incomplete_features))]

trait Foo {}

impl<const N: usize> Foo for [(); N]
    where
        Self:FooImpl<{N==0}>
//[full]~^ERROR constant expression depends on a generic parameter
//[min]~^^ERROR generic parameters may not be used in const operations
{}

trait FooImpl<const IS_ZERO: bool>{}

impl FooImpl<true> for [(); 0] {}

impl<const N:usize> FooImpl<false> for [();N] {}

fn foo(_: impl Foo) {}

fn main() {
    foo([]);
    foo([()]);
}
