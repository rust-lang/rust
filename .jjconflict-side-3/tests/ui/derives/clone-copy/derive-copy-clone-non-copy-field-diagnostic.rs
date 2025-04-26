//! Regression test for <https://github.com/rust-lang/rust/issues/27340>.
//! Test anon fields in tuple-syntax structs which don't implement trait
//! get nice error message mentioning the type of field and its span.

struct Foo;
#[derive(Copy, Clone)]
struct Bar(Foo);
//~^ ERROR: the trait `Copy` cannot be implemented for this type
//~| ERROR: `Foo: Clone` is not satisfied

fn main() {}
