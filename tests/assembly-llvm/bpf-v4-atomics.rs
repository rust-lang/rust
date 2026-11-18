//@ add-minicore
//@ assembly-output: emit-asm
//@ compile-flags: --target bpfel-unknown-none -Ctarget-cpu=v4
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

// CHECK-LABEL: atomic_load_relaxed8:
// CHECK: w0 = *(u8 *)(r1 + 0)
// CHECK: exit
#[no_mangle]
pub fn atomic_load_relaxed8(p: *const u8) -> u8 {
    unsafe { atomic::atomic_load::<u8, { AtomicOrdering::Relaxed }, false>(p) }
}

// CHECK-LABEL: atomic_load_acquire8:
// CHECK: w0 = load_acquire((u8 *)(r1 + 0))
// CHECK: exit
#[no_mangle]
pub fn atomic_load_acquire8(p: *const u8) -> u8 {
    unsafe { atomic::atomic_load::<u8, { AtomicOrdering::Acquire }, false>(p) }
}

// CHECK-LABEL: atomic_store_relaxed8:
// CHECK: *(u8 *)(r1 + 0) = w2
// CHECK: exit
#[no_mangle]
pub fn atomic_store_relaxed8(p: *mut u8, v: u8) {
    unsafe { atomic::atomic_store::<u8, { AtomicOrdering::Relaxed }, false>(p, v) }
}

// CHECK-LABEL: atomic_store_release8:
// CHECK: store_release((u8 *)(r1 + 0), w2)
// CHECK: exit
#[no_mangle]
pub fn atomic_store_release8(p: *mut u8, v: u8) {
    unsafe { atomic::atomic_store::<u8, { AtomicOrdering::Release }, false>(p, v) }
}

// CHECK-LABEL: atomic_load_relaxed16:
// CHECK: w0 = *(u16 *)(r1 + 0)
// CHECK: exit
#[no_mangle]
pub fn atomic_load_relaxed16(p: *const u16) -> u16 {
    unsafe { atomic::atomic_load::<u16, { AtomicOrdering::Relaxed }, false>(p) }
}

// CHECK-LABEL: atomic_load_acquire16:
// CHECK: w0 = load_acquire((u16 *)(r1 + 0))
// CHECK: exit
#[no_mangle]
pub fn atomic_load_acquire16(p: *const u16) -> u16 {
    unsafe { atomic::atomic_load::<u16, { AtomicOrdering::Acquire }, false>(p) }
}

// CHECK-LABEL: atomic_store_relaxed16:
// CHECK: *(u16 *)(r1 + 0) = w2
// CHECK: exit
#[no_mangle]
pub fn atomic_store_relaxed16(p: *mut u16, v: u16) {
    unsafe { atomic::atomic_store::<u16, { AtomicOrdering::Relaxed }, false>(p, v) }
}

// CHECK-LABEL: atomic_store_release16:
// CHECK: store_release((u16 *)(r1 + 0), w2)
// CHECK: exit
#[no_mangle]
pub fn atomic_store_release16(p: *mut u16, v: u16) {
    unsafe { atomic::atomic_store::<u16, { AtomicOrdering::Release }, false>(p, v) }
}

// CHECK-LABEL: atomic_load_relaxed32:
// CHECK: w0 = *(u32 *)(r1 + 0)
// CHECK: exit
#[no_mangle]
pub fn atomic_load_relaxed32(p: *const u32) -> u32 {
    unsafe { atomic::atomic_load::<u32, { AtomicOrdering::Relaxed }, false>(p) }
}

// CHECK-LABEL: atomic_load_acquire32:
// CHECK: w0 = load_acquire((u32 *)(r1 + 0))
// CHECK: exit
#[no_mangle]
pub fn atomic_load_acquire32(p: *const u32) -> u32 {
    unsafe { atomic::atomic_load::<u32, { AtomicOrdering::Acquire }, false>(p) }
}

// CHECK-LABEL: atomic_store_relaxed32:
// CHECK: *(u32 *)(r1 + 0) = w2
// CHECK: exit
#[no_mangle]
pub fn atomic_store_relaxed32(p: *mut u32, v: u32) {
    unsafe { atomic::atomic_store::<u32, { AtomicOrdering::Relaxed }, false>(p, v) }
}

// CHECK-LABEL: atomic_store_release32:
// CHECK: store_release((u32 *)(r1 + 0), w2)
// CHECK: exit
#[no_mangle]
pub fn atomic_store_release32(p: *mut u32, v: u32) {
    unsafe { atomic::atomic_store::<u32, { AtomicOrdering::Release }, false>(p, v) }
}
