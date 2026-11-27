// Should not rely on the aliasing model for its failure.
//@compile-flags: -Zmiri-disable-stacked-borrows

use std::sync::atomic::{AtomicI32, Ordering};

fn main() {
    static X: i32 = 0;
    let x = &X as *const i32 as *const AtomicI32;
    let x = unsafe { &*x };
    // Some targets can implement atomic loads via compare_exchange, so we cannot allow them on
    // read-only memory.
    x.load(Ordering::Acquire); //~ERROR: cannot be performed on read-only memory
}
