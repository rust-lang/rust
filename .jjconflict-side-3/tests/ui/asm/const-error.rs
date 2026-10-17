//@ only-x86_64
//@ needs-asm-support
//@ check-pass

// Test to make sure that we emit const errors late for inline asm,
// which is consistent with inline const blocks.

use std::arch::asm;

fn test<T>() {
    unsafe {
        // No error here, as this does not get monomorphized.
        asm!("/* {} */", const 1 / 0);
    }
}

fn main() {}
