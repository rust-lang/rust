//@ ignore-backends: gcc
// FIXME: linking on windows (specifically mingw) not yet supported, see tracking issue #125418
//@ ignore-windows
// Tests whether EIIs work on statics
#![feature(extern_item_impls)]

#[eii(hello)]
static HELLO: u64;

#[hello]
//~^ ERROR `#[hello]` must be used on a static
fn hello_impl() -> u64 {
    5
}

fn main() { }
