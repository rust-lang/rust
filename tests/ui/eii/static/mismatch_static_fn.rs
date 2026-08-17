//@ ignore-backends: gcc
// FIXME(#125418): linking on Windows GNU targets is not yet supported.
//@ ignore-windows-gnu
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
