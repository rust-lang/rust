//@ only-aarch64
//@ assembly-output: emit-asm
//@ revisions: inline outline
//@ compile-flags: -Copt-level=2
//@[inline] compile-flags: -Ctarget-feature=-outline-atomics

// Without the `outline-atomics` feature, atomics must not call the libgcc `__aarch64_*` helpers,
// which are not available on targets like the Linux kernel.

#![feature(integer_atomics)]
#![crate_type = "lib"]

use std::sync::atomic::{AtomicU32, AtomicU128, Ordering};

// inline-NOT: __aarch64_
// outline-DAG: bl __aarch64_ldadd4_acq_rel
// outline-DAG: bl __aarch64_cas16_sync

#[unsafe(no_mangle)]
pub fn fetch_add_u32(atomic: &AtomicU32, value: u32) -> u32 {
    atomic.fetch_add(value, Ordering::SeqCst)
}

#[unsafe(no_mangle)]
pub fn compare_exchange_u128(atomic: &AtomicU128, current: u128, new: u128) -> bool {
    atomic.compare_exchange(current, new, Ordering::SeqCst, Ordering::SeqCst).is_ok()
}
