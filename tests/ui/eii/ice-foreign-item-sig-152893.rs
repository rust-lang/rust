// Regression test for #152893
// This used to ICE with "foreign item sig" when using #[eii]
// on a function that shadows a struct name.
// Now it correctly reports a name redefinition error.

#![feature(extern_item_impls)]

struct Foo;

#[eii]
pub fn Foo(x: u64) {}
//~^ ERROR the name `Foo` is defined multiple times

fn main() {}
