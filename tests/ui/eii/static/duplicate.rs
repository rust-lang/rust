//@ ignore-backends: gcc
// FIXME(#125418): linking on Windows GNU targets is not yet supported.
//@ ignore-windows-gnu
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
