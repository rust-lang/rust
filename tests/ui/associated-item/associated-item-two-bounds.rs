// This test is a regression test for #34792

//@ check-pass

pub struct A;
pub struct B;

pub trait Foo {
    type T: PartialEq<A> + PartialEq<B>;
}

pub fn generic<F: Foo>(t: F::T, a: A, b: B) -> bool {
    t == a && t == b
}

pub fn main() {}
