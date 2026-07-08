//! Regression test for <https://github.com/rust-lang/rust/issues/36299>.
//! This used to ICE.

struct Foo<'a, A> {}
//~^ ERROR parameter `'a` is never used
//~| ERROR parameter `A` is never used

fn main() {}
