// Stacked Borrows doesn't like this.
//@compile-flags: -Zmiri-tree-borrows

use std::sync::atomic::*;

fn main() {
    // Atomic loads from read-only memory are fine if they are relaxed and small.
    static X: i32 = 0;
    let x = &X as *const i32 as *const AtomicI32;
    let x = unsafe { &*x };
    x.load(Ordering::Relaxed);
}
