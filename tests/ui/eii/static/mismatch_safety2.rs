//@ ignore-backends: gcc
// FIXME(#125418): linking on Windows GNU targets is not yet supported.
//@ ignore-windows-gnu
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
