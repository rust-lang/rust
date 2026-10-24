//@ run-pass
#![feature(generic_const_exprs)]
#![allow(incomplete_features)]

trait Foo {}

impl<const N: usize> Foo for [(); N] where Self: FooImpl<{ N == 0 }> {}

trait FooImpl<const IS_ZERO: bool> {}

impl FooImpl<{ 0u8 == 0u8 }> for [(); 0] {}

impl<const N: usize> FooImpl<{ 0u8 != 0u8 }> for [(); N] {}

fn foo<T: Foo>(_v: T) {}

fn main() {
    foo([]);
    foo([()]);
}
