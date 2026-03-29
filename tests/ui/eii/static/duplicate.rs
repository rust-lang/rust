//@ ignore-backends: gcc
// FIXME: linking on windows (speciifcally mingw) not yet supported, see tracking issue #125418
//@ ignore-windows
// Tests whether EIIs work on statics
#![feature(extern_item_impls)]

#[eii(hello)]
static HELLO: u64;

#[hello]
static HELLO_IMPL1: u64 = 5;
//~^ ERROR multiple implementations of `#[hello]`

#[hello]
static HELLO_IMPL2: u64 = 6;

// what you would write:
fn main() {
    // directly
    println!("{HELLO_IMPL1}");
    println!("{HELLO_IMPL2}");

    // through the alias
    println!("{HELLO}");
}
