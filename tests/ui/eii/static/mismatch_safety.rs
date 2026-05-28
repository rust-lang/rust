//@ ignore-backends: gcc
// FIXME: linking on windows (specifically mingw) not yet supported, see tracking issue #125418
//@ ignore-windows
// Tests whether EIIs work on statics
#![feature(extern_item_impls)]

#[eii(hello)]
unsafe static HELLO: u64;

#[hello]
//~^ ERROR safety does not match with the definition of`#[hello]`
static HELLO_IMPL: u64 = 5;

// what you would write:
fn main() {
    // directly
    println!("{HELLO_IMPL}");

    // through the alias
    println!("{}", unsafe { HELLO });
}
