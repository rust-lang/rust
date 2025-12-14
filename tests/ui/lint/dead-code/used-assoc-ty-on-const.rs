//@ check-pass

#![deny(dead_code)]

pub struct A;

trait B {
    type Assoc;
}

impl B for A {
    type Assoc = A;
}

const X: <A as B>::Assoc = A;

fn main() {
    let _x: A = X;
}
