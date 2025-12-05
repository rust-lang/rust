// Should not rely on the aliasing model for its failure.
//@compile-flags: -Zmiri-disable-stacked-borrows

use std::sync::atomic::{AtomicI32, Ordering};

fn main() {
    static X: i32 = 0;
    let x = &X as *const i32 as *const AtomicI32;
    let x = unsafe { &*x };
    x.compare_exchange(1, 2, Ordering::Relaxed, Ordering::Relaxed).unwrap_err(); //~ERROR: cannot be performed on read-only memory
}
