// Test that an unaligned `MaybeDangling<&u8>` is still detected as UB.
//
//@compile-flags: -Zmiri-disable-stacked-borrows
#![feature(maybe_dangling)]

use std::mem::{MaybeDangling, transmute};

fn main() {
    let a = [1u16, 0u16];
    unsafe {
        let unaligned = MaybeDangling::new(a.as_ptr().byte_add(1));
        transmute::<MaybeDangling<*const u16>, MaybeDangling<&u16>>(unaligned)
        //~^ ERROR: Undefined Behavior: constructing invalid value: encountered an unaligned reference
    };
}
