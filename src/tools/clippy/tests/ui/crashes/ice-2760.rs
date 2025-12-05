//@ check-pass

#![allow(
    unused_variables,
    clippy::disallowed_names,
    clippy::needless_pass_by_value,
    dead_code
)]

// This should not compile-fail with:
//
//      error[E0277]: the trait bound `T: Foo` is not satisfied
// See https://github.com/rust-lang/rust-clippy/issues/2760

trait Foo {
    type Bar;
}

struct Baz<T: Foo> {
    bar: T::Bar,
}

fn take<T: Foo>(baz: Baz<T>) {}

fn main() {}
