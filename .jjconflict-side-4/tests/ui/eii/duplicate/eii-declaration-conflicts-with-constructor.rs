#![feature(extern_item_impls)]

// Regression test for <https://github.com/rust-lang/rust/issues/153502>:
// Regression test for <https://github.com/rust-lang/rust/issues/152893>:

struct Foo(i32);
struct Bar;

#[eii]
pub fn Foo(x: u64) {}
//~^ ERROR the name `Foo` is defined multiple times

#[eii]
pub fn Bar(x: u64) {}
//~^ ERROR the name `Bar` is defined multiple times

fn main() {}
