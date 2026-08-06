// Ensure `memchr` communicates that a returned index is in bounds.
//@ compile-flags: -Copt-level=3 -Zinline-mir=false
//@ only-x86_64
//@ revisions: llvm-old llvm-new
//@ [llvm-old] max-llvm-major-version: 21
//@ [llvm-new] min-llvm-version: 22

#![crate_type = "lib"]
#![feature(slice_internals)]

extern crate core;

use core::slice::memchr::{memchr, memrchr};

// CHECK-LABEL: @find_char
#[no_mangle]
pub fn find_char(haystack: &str, needle: char) -> Option<usize> {
    // llvm-old: call void @llvm.assume
    // llvm-new-NOT: phi { i64, i64 }
    // CHECK: ret { i64, i64 }
    haystack.find(needle)
}

// CHECK-LABEL: @find_byte
#[no_mangle]
pub fn find_byte(haystack: &[u8], needle: u8) -> Option<u8> {
    // llvm-new-NOT: panic_bounds_check
    // CHECK: ret { i1, i8 }
    memchr(needle, haystack).map(|index| haystack[index])
}

// CHECK-LABEL: @rfind_byte
#[no_mangle]
pub fn rfind_byte(haystack: &[u8], needle: u8) -> Option<u8> {
    // CHECK-NOT: panic_bounds_check
    // CHECK: ret { i1, i8 }
    memrchr(needle, haystack).map(|index| haystack[index])
}
