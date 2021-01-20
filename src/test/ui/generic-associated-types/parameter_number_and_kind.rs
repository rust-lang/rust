#![allow(incomplete_features)]
#![feature(generic_associated_types)]
#![feature(associated_type_defaults)]

trait Foo {
    type A<'a>;
    type B<'a, 'b>;
    type C;
    type D<T>;
    type E<'a, T>;
    // Test parameters in default values
    type FOk<T> = Self::E<'static, T>;
    type FErr1 = Self::E<'static, 'static>;
    //~^ ERROR wrong number of lifetime arguments: expected 1, found 2
    //~| ERROR wrong number of type arguments: expected 1, found 0
    type FErr2<T> = Self::E<'static, T, u32>;
    //~^ ERROR wrong number of type arguments: expected 1, found 2
}

fn main() {}
