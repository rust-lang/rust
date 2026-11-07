//@ add-minicore
//@ assembly-output: emit-asm
//@ compile-flags: --target bpfel-unknown-none -C target_feature=+allows-misaligned-mem-access
//@ min-llvm-version: 22
//@ needs-llvm-components: bpf
#![feature(no_core)]
#![crate_type = "rlib"]
#![no_core]

extern crate minicore;
use minicore::*;

// CHECK-LABEL: test_load_i64:
// CHECK:       r0 = *(u64 *)(r1 + 0)
#[no_mangle]
pub unsafe fn test_load_i64(p: *const u64) -> u64 {
    let mut tmp: u64 = 0;
    copy_nonoverlapping(p as *const u8, &mut tmp as *mut u64 as *mut u8, 8);
    tmp
}

// CHECK-LABEL: test_store_i64:
// CHECK:       *(u64 *)(r1 + 0) = r2
#[no_mangle]
pub unsafe fn test_store_i64(p: *mut u64, v: u64) {
    copy_nonoverlapping(&v as *const u64 as *const u8, p as *mut u8, 8);
}
