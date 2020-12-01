#![allow(
    unused_variables,
    clippy::blacklisted_name,
    clippy::needless_pass_by_value,
    dead_code
)]

/// This should not compile-fail with:
///
///      error[E0277]: the trait bound `T: Foo` is not satisfied
// See rust-lang/rust-clippy#2760.

trait Foo {
    type Bar;
}

struct Baz<T: Foo> {
    bar: T::Bar,
}

fn take<T: Foo>(baz: Baz<T>) {}

fn main() {}
