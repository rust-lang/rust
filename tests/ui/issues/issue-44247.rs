//@ check-pass
#![allow(dead_code)]
trait T {
    type X;
    const X: Self::X;
}
fn foo<X: T>() {
    let _: X::X = X::X;
}

trait S {
    const X: Self::X;
    type X;
}
fn bar<X: S>() {
    let _: X::X = X::X;
}

fn main() {}
