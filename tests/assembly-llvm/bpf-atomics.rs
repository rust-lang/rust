//@ add-minicore
//@ assembly-output: emit-asm
//@ compile-flags: --target bpfel-unknown-none
//@ needs-llvm-components: bpf

// Regression test for 64-bit atomic load/store on BPF. The BPF ISA *does* have
// atomic instructions and LLVM can lower them, but it only selects the 64-bit
// atomic load/store patterns (relaxed included, which lower to plain loads and
// stores) when the subtarget is the `v4` CPU. The target spec pins `cpu = "v4"`,
// so `core::io::Error`'s 64-bit `AtomicPtr` relaxed load/store (the path that
// used to fail at ISel with `Cannot select: ... AtomicLoad`) must lower here.
#![feature(no_core)]
#![crate_type = "rlib"]
#![no_core]

extern crate minicore;
use minicore::*;

// Relaxed load/store lower to plain loads/stores (no acquire/release fence).
// CHECK-LABEL: atomic_load_relaxed:
// CHECK: r0 = *(u64 *)(r1 + 0)
// CHECK: exit
#[no_mangle]
pub fn atomic_load_relaxed(p: *const u64) -> u64 {
    unsafe { atomic::atomic_load::<u64, { AtomicOrdering::Relaxed }, false>(p) }
}

// CHECK-LABEL: atomic_store_relaxed:
// CHECK: *(u64 *)(r1 + 0) = r2
// CHECK: exit
#[no_mangle]
pub fn atomic_store_relaxed(p: *mut u64, v: u64) {
    unsafe { atomic::atomic_store::<u64, { AtomicOrdering::Relaxed }, false>(p, v) }
}

// Acquire load lowers to the `load_acquire` BPF instruction.
// CHECK-LABEL: atomic_load_acquire:
// CHECK: r0 = load_acquire((u64 *)(r1 + 0))
// CHECK: exit
#[no_mangle]
pub fn atomic_load_acquire(p: *const u64) -> u64 {
    unsafe { atomic::atomic_load::<u64, { AtomicOrdering::Acquire }, false>(p) }
}

// Release store lowers to the `store_release` BPF instruction.
// CHECK-LABEL: atomic_store_release:
// CHECK: store_release((u64 *)(r1 + 0), r2)
// CHECK: exit
#[no_mangle]
pub fn atomic_store_release(p: *mut u64, v: u64) {
    unsafe { atomic::atomic_store::<u64, { AtomicOrdering::Release }, false>(p, v) }
}
