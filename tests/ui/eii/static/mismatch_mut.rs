//@ ignore-backends: gcc
// FIXME(#125418): linking on Windows GNU targets is not yet supported.
//@ ignore-windows-gnu
// Tests whether EIIs work on statics
#![feature(extern_item_impls)]

#[eii(hello)]
//~^ ERROR `#[eii]` cannot be used on mutable statics
static mut HELLO: u64;

#[hello]
//~^ ERROR mutability does not match with the definition of`#[hello]`
static HELLO_IMPL: u64 = 5;

// what you would write:
fn main() {
    // directly
    println!("{HELLO_IMPL}");

    // through the alias
    println!("{}", unsafe { HELLO });
}
