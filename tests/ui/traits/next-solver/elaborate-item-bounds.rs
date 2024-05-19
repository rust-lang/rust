//@ compile-flags: -Znext-solver
//@ check-pass

trait Foo {
    type Bar: Bar;
}

trait Bar: Baz {}

trait Baz {}

fn main() {}
