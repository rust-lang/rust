//@ ignore-backends: gcc
// FIXME(#125418): linking on Windows GNU targets is not yet supported.
//@ ignore-windows-gnu
// Tests whether calling EIIs works with the declaration in the same crate.
#![feature(extern_item_impls)]

#[eii(inline)]
//~^ ERROR `#[inline]` function required, but not found
fn test(x: u64);

#[inline]
//~^ ERROR `inline` is ambiguous
fn test_impl(x: u64) {
    println!("{x:?}")
}

fn main() {}
