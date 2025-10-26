//@ assembly-output: emit-asm
//@ compile-flags: -Copt-level=3
//@ only-aarch64
//@ only-linux

#![crate_type = "rlib"]

use std::sync::atomic::AtomicI32;
use std::sync::atomic::Ordering::*;

// Verify config on outline-atomics works (it is always enabled on aarch64-linux).
#[cfg(not(target_feature = "outline-atomics"))]
compile_error!("outline-atomics is not enabled");

pub fn compare_exchange(a: &AtomicI32) {
    // On AArch64 LLVM should outline atomic operations.
    // CHECK: __aarch64_cas4_relax
    let _ = a.compare_exchange(0, 10, Relaxed, Relaxed);
}
