//@ check-pass

#![deny(dead_code)]

pub struct A;

trait B {
    type Assoc;
}

impl B for A {
    type Assoc = A;
}

trait C {}

impl C for <A as B>::Assoc {}

fn foo<T: C>() {}

fn main() {
    foo::<A>();
}
