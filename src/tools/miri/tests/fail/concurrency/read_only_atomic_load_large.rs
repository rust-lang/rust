// Should not rely on the aliasing model for its failure.
//@compile-flags: -Zmiri-disable-stacked-borrows
// Needs atomic accesses larger than the pointer size
//@ignore-bitwidth: 64
//@ignore-target: mips-

use std::sync::atomic::{AtomicI64, Ordering};

#[repr(align(8))]
struct AlignedI64(#[allow(dead_code)] i64);

fn main() {
    static X: AlignedI64 = AlignedI64(0);
    let x = &X as *const AlignedI64 as *const AtomicI64;
    let x = unsafe { &*x };
    // Some targets can implement atomic loads via compare_exchange, so we cannot allow them on
    // read-only memory.
    x.load(Ordering::Relaxed); //~ERROR: cannot be performed on read-only memory
}
