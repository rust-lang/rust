//@ ignore-backends: gcc
// FIXME: linking on windows (specifically mingw) not yet supported, see tracking issue #125418
//@ ignore-windows
// Tests that mismatching types of the declaration and definition are rejected
#![feature(extern_item_impls)]

use std::ptr;

#[eii(hello)]
static HELLO: for<'a> fn(&'a u8) -> &'static u8;

#[hello]
static HELLO_IMPL: for<'a> fn(&'a u8) -> &'a u8 = |_| todo!();
//~^ ERROR mismatched types

fn main() {
}
