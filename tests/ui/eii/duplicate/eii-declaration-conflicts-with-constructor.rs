#![feature(extern_item_impls)]

// Regression test for <https://github.com/rust-lang/rust/issues/153502>:

struct Foo(i32);

#[eii]
pub fn Foo(x: u64) {}
//~^ ERROR the name `Foo` is defined multiple times

fn main() {}
