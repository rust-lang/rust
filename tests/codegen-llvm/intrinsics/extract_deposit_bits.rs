//@ compile-flags: -C no-prepopulate-passes -C opt-level=3
//@ revisions: LLVM22 LLVM23
//@ [LLVM22] max-llvm-major-version: 22
//@ [LLVM23] min-llvm-version: 23

#![crate_type = "lib"]
#![feature(core_intrinsics, uint_gather_scatter_bits)]

use std::intrinsics::{deposit_bits, extract_bits};

// CHECK-LABEL: @extract_bits_u32
#[no_mangle]
pub fn extract_bits_u32(x: u32, mask: u32) -> u32 {
    // LLVM23: call i32 @llvm.pext.i32(i32 {{.*}}, i32 {{.*}})
    // LLVM22-NOT: @llvm.pext
    // LLVM22: call {{.*}}extract_bits
    extract_bits(x, mask)
}

// CHECK-LABEL: @deposit_bits_u32
#[no_mangle]
pub fn deposit_bits_u32(x: u32, mask: u32) -> u32 {
    // LLVM23: call i32 @llvm.pdep.i32(i32 {{.*}}, i32 {{.*}})
    // LLVM22-NOT: @llvm.pdep
    // LLVM22: call {{.*}}deposit_bits
    deposit_bits(x, mask)
}

// CHECK-LABEL: @extract_bits_u64
#[no_mangle]
pub fn extract_bits_u64(x: u64, mask: u64) -> u64 {
    // LLVM23: call i64 @llvm.pext.i64(i64 {{.*}}, i64 {{.*}})
    // LLVM22-NOT: @llvm.pext
    // LLVM22: call {{.*}}extract_bits
    extract_bits(x, mask)
}

// CHECK-LABEL: @deposit_bits_u64
#[no_mangle]
pub fn deposit_bits_u64(x: u64, mask: u64) -> u64 {
    // LLVM23: call i64 @llvm.pdep.i64(i64 {{.*}}, i64 {{.*}})
    // LLVM22-NOT: @llvm.pdep
    // LLVM22: call {{.*}}deposit_bits
    deposit_bits(x, mask)
}
