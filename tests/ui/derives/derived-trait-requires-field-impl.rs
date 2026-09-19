//! Regression test for https://github.com/rust-lang/rust/issues/27340
struct Foo;
#[derive(Copy, Clone)]
struct Bar(Foo);
//~^ ERROR: the trait `Copy` cannot be implemented for this type
//~| ERROR: `Foo: Clone` is not satisfied

fn main() {}
