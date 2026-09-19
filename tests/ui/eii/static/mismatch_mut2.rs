//@ ignore-backends: gcc
// FIXME(#125418): linking on Windows GNU targets is not yet supported.
//@ ignore-windows-gnu
// Tests whether EIIs work on statics
#![feature(extern_item_impls)]

#[eii(hello)]
static HELLO: u64;

#[hello]
//~^ ERROR mutability does not match with the definition of`#[hello]`
static mut HELLO_IMPL: u64 = 5;

// what you would write:
fn main() {
    // directly
    println!("{}", unsafe { HELLO_IMPL });

    // through the alias
    println!("{HELLO}");
}
