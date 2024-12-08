// Make sure we find these even with many checks disabled.
//@compile-flags: -Zmiri-disable-alignment-check -Zmiri-disable-stacked-borrows -Zmiri-disable-validation
#![feature(core_intrinsics)]
#![feature(custom_mir)]

use std::intrinsics::mir::*;

#[custom_mir(dialect = "runtime", phase = "optimized")]
fn cast(ptr: *const char) -> u32 {
    mir! {
        {
            RET = *ptr as u32; //~ERROR: interpreting an invalid 32-bit value as a char
            Return()
        }
    }
}

pub fn main() {
    let v = u32::MAX;
    cast(&v as *const u32 as *const char);
}
