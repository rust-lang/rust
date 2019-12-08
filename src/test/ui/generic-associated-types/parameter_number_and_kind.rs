#![allow(incomplete_features)]
#![feature(generic_associated_types)]
#![feature(associated_type_defaults)]

trait Foo {
    type A<'a>;
    type B<'a, 'b>;
    type C;
    type D<T>;
    //~^ ERROR type-generic associated types are not yet implemented
    type E<'a, T>;
    //~^ ERROR type-generic associated types are not yet implemented
    // Test parameters in default values
    type FOk<T> = Self::E<'static, T>;
    //~^ ERROR type-generic associated types are not yet implemented
    type FErr1 = Self::E<'static, 'static>;
    //~^ ERROR wrong number of lifetime arguments: expected 1, found 2
    //~| ERROR wrong number of type arguments: expected 1, found 0
    type FErr2<T> = Self::E<'static, T, u32>;
    //~^ ERROR type-generic associated types are not yet implemented
    //~| ERROR wrong number of type arguments: expected 1, found 2
}

fn main() {}
