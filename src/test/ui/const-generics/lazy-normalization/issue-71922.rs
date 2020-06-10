#![feature(const_generics)]
//~^ WARN the feature `const_generics` is incomplete
trait Foo {}

impl<const N: usize> Foo for [(); N] where Self: FooImpl<{ N == 0 }> {}
//~^ ERROR constant expression depends on a generic parameter

trait FooImpl<const IS_ZERO: bool> {}

impl FooImpl<{ 0u8 == 0u8 }> for [(); 0] {}

impl<const N: usize> FooImpl<{ 0u8 != 0u8 }> for [(); N] {}

fn foo<T: Foo>(_: T) {}

fn main() {
    foo([]);
    foo([()]);
}
