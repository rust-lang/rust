//@ ignore-backends: gcc
// FIXME: linking on windows (speciifcally mingw) not yet supported, see tracking issue #125418
//@ ignore-windows
// Tests whether EIIs work on statics
#![feature(extern_item_impls)]

#[eii(hello)]
fn hello() -> u64;

#[hello]
//~^ ERROR `#[hello]` must be used on a function
static HELLO_IMPL: u64 = 5;

fn main() { }
