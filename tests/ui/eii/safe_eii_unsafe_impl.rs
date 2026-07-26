// Tests that safe EIIs reject `unsafe(...)` implementation attributes.
#![feature(extern_item_impls)]

#[eii]
fn foo(x: u64) -> u64;

#[unsafe(foo)] //~ ERROR `foo` is not unsafe to implement
fn foo_impl(x: u64) -> u64 {
    x
}

fn main() {
    foo(0);
}
