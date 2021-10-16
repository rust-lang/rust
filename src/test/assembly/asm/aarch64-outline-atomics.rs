// min-llvm-version: 12.0
// assembly-output: emit-asm
// compile-flags: -O
// compile-flags: --target aarch64-unknown-linux-gnu
// needs-llvm-components: aarch64
// only-aarch64
// only-linux

#![crate_type = "rlib"]

use std::sync::atomic::{AtomicI32, Ordering::*};

pub fn compare_exchange(a: &AtomicI32) {
    // On AArch64 LLVM should outline atomic operations.
    // CHECK: __aarch64_cas4_relax
    let _ = a.compare_exchange(0, 10, Relaxed, Relaxed);
}
