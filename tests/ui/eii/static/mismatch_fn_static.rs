//@ ignore-backends: gcc
// FIXME(#125418): linking on Windows GNU targets is not yet supported.
//@ ignore-windows-gnu
// Tests whether EIIs work on statics
#![feature(extern_item_impls)]

#[eii(hello)]
fn hello() -> u64;

#[hello]
//~^ ERROR `#[hello]` must be used on a function
static HELLO_IMPL: u64 = 5;

fn main() { }
