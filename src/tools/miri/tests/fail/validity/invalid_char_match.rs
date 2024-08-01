// Make sure we find these even with many checks disabled.
//@compile-flags: -Zmiri-disable-alignment-check -Zmiri-disable-stacked-borrows -Zmiri-disable-validation
#![feature(core_intrinsics)]
#![feature(custom_mir)]

use std::intrinsics::mir::*;

#[custom_mir(dialect = "runtime", phase = "optimized")]
fn switch_int(ptr: *const char) {
    mir! {
        {
            match *ptr { //~ERROR: interpreting an invalid 32-bit value as a char
                '0' => ret,
                _ => ret,
            }
        }
        ret = {
            Return()
        }
    }
}

pub fn main() {
    let v = u32::MAX;
    switch_int(&v as *const u32 as *const char);
}
