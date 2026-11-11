//@ check-pass

#![allow(unused)]

trait Foo {
    type Bar;
    type Baz: Bar<Self::Bar>;
}

trait Bar<T> {}

fn x<T: Foo<Bar = U>, U>(t: &T) {}

fn main() {}
