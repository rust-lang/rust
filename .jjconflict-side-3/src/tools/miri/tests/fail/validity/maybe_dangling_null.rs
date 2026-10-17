// Test that a null `MaybeDangling<&u8>` is still detected as UB.
//
//@compile-flags: -Zmiri-disable-stacked-borrows
#![feature(maybe_dangling)]

use std::mem::{MaybeDangling, transmute};
use std::ptr::null;

fn main() {
    let null = MaybeDangling::new(null());
    unsafe { transmute::<MaybeDangling<*const u8>, MaybeDangling<&u8>>(null) };
    //~^ ERROR: encountered a null reference
}
