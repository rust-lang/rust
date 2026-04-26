//@ ignore-backends: gcc
// FIXME: linking on windows (specifically mingw) not yet supported, see tracking issue #125418
//@ ignore-windows
// Tests whether EIIs work on statics
#![feature(extern_item_impls)]

#[eii(hello)]
static HELLO: u64;

#[hello]
unsafe static HELLO_IMPL: u64 = 5;
//~^ ERROR static items cannot be declared with `unsafe` safety qualifier outside of `extern` block

// what you would write:
fn main() {
    // directly
    println!("{HELLO_IMPL}");

    // through the alias
    println!("{}", unsafe { HELLO });
}
