//@ ignore-backends: gcc
// FIXME: linking on windows (speciifcally mingw) not yet supported, see tracking issue #125418
//@ ignore-windows
// Tests whether calling EIIs works with the declaration in the same crate.
#![feature(extern_item_impls)]

#[eii(inline)]
//~^ ERROR `#[inline]` required, but not found
fn test(x: u64);

#[inline]
//~^ ERROR `inline` is ambiguous
fn test_impl(x: u64) {
    println!("{x:?}")
}

fn main() { }
