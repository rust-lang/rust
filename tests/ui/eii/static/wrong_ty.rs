//@ ignore-backends: gcc
// FIXME(#125418): linking on Windows GNU targets is not yet supported.
//@ ignore-windows-gnu
// Tests that mismatching types of the declaration and definition are rejected
#![feature(extern_item_impls)]

use std::ptr;

#[eii(hello)]
static HELLO: u64;

#[hello]
static HELLO_IMPL: bool = true;
//~^ ERROR static `HELLO_IMPL` has a type that is incompatible with the declaration of `#[hello]` [E0806]

fn main() {

}
