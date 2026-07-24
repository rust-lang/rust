// Regression test for #52560: point at an imperfect derive when its generated
// bound is unsatisfied.

use std::fmt::Debug;

#[derive(Debug)]
struct Foo<B: Bar>(B::Item);

trait Bar {
    type Item: Debug;
}

fn print<B: Bar>(value: Foo<B>) {
    println!("{value:?}"); //~ ERROR `B` doesn't implement `Debug`
}

fn main() {}
