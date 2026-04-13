//@ ignore-backends: gcc
// FIXME: linking on windows (specifically mingw) not yet supported, see tracking issue #125418
//@ ignore-windows
// Tests whether EIIs work on statics
#![feature(extern_item_impls)]

#[eii]
//~^ ERROR `#[eii]` requires the name as an explicit argument when used on a static
static HELLO: u64;

fn main() { }
