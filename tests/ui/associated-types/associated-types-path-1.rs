// Test that we have one and only one associated type per ref.

pub trait Foo {
    type A;
}
pub trait Bar {
    type A;
}

pub fn f1<T>(a: T, x: T::A) {} //~ERROR associated type `A` not found
pub fn f2<T: Foo + Bar>(a: T, x: T::A) {} //~ERROR ambiguous associated type `A`

pub fn main() {}
