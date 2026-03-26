//@ run-pass
//@ check-run-results
//@ ignore-backends: gcc
// FIXME: linking on windows (speciifcally mingw) not yet supported, see tracking issue #125418
//@ ignore-windows
// Tests whether EIIs work on statics
#![feature(extern_item_impls)]

#[eii(hello)]
static HELLO: u64;

#[hello]
static HELLO_IMPL: u64 = 5;

// what you would write:
fn main() {
    // directly
    println!("{HELLO_IMPL}");

    // through the alias
    println!("{HELLO}");
}
