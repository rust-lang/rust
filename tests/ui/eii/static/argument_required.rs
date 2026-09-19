//@ ignore-backends: gcc
// FIXME(#125418): linking on Windows GNU targets is not yet supported.
//@ ignore-windows-gnu
// Tests whether EIIs work on statics
#![feature(extern_item_impls)]

#[eii]
//~^ ERROR `#[eii]` requires the name as an explicit argument when used on a static
static HELLO: u64;

fn main() { }
