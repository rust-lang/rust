//@ add-minicore
//@ min-llvm-version: 22
//@ assembly-output: emit-asm
//@ compile-flags: --target thumbv8m.main-none-eabi --crate-type lib -Copt-level=1
//@ needs-llvm-components: arm
#![crate_type = "lib"]
#![feature(abi_cmse_nonsecure_call, cmse_nonsecure_entry, no_core, lang_items)]
#![no_core]

// Test that padding and other uninitialized bytes are zeroed when a value crosses the secure
// boundary.
//
// The assembly uses the following instructions for clearing the bits:
//
// - `uxtb` clears bits 8..32
// - `uxth` clears bits 16..32
// - `bic` clears bits based on a mask

extern crate minicore;
use minicore::*;

#[repr(C)]
pub struct InnerPadding {
    a: u8,
    b: u16,
}

// CHECK-LABEL: c_ret_with_inner_padding:
// CHECK: mov r7, sp
// CHECK-NEXT: orr.w r0, r0, r1, lsl #16
#[no_mangle]
pub extern "C" fn c_ret_with_inner_padding(a: u8, b: u16) -> InnerPadding {
    InnerPadding { a, b }
}

// CHECK-LABEL: cmse_ret_with_inner_padding:
// CHECK: mov r7, sp
// CHECK-NEXT: uxtb r0, r0
// CHECK-NEXT: orr.w r0, r0, r1, lsl #16
#[no_mangle]
pub extern "cmse-nonsecure-entry" fn cmse_ret_with_inner_padding(a: u8, b: u16) -> InnerPadding {
    InnerPadding { a, b }
}

#[repr(C)]
pub struct TrailingPadding {
    a: u16,
    b: u8,
}

// CHECK-LABEL: c_ret_with_trailing_padding:
// CHECK: mov r7, sp
// CHECK-NEXT: orr.w r0, r0, r1, lsl #16
#[no_mangle]
pub extern "C" fn c_ret_with_trailing_padding(a: u16, b: u8) -> TrailingPadding {
    TrailingPadding { a, b }
}

// CHECK-LABEL: cmse_ret_with_trailing_padding:
// CHECK: mov r7, sp
// CHECK-NEXT: uxtb r1, r1
// CHECK-NEXT: uxth r0, r0
// CHECK-NEXT: orr.w r0, r0, r1, lsl #16
#[no_mangle]
pub extern "cmse-nonsecure-entry" fn cmse_ret_with_trailing_padding(
    a: u16,
    b: u8,
) -> TrailingPadding {
    TrailingPadding { a, b }
}

#[repr(C, align(2))]
pub struct WideU8 {
    a: u8,
}

// CHECK-LABEL: c_ret_with_wide_u8:
// CHECK: mov r7, sp
// CHECK-NEXT: orr.w r0, r0, r1, lsl #16
#[no_mangle]
pub extern "C" fn c_ret_with_wide_u8(a: u8, b: u8) -> [WideU8; 2] {
    [WideU8 { a }, WideU8 { a: b }]
}

// CHECK-LABEL: cmse_ret_with_wide_u8:
// CHECK: mov r7, sp
// CHECK-NEXT: uxtb r1, r1
// CHECK-NEXT: uxtb r0, r0
// CHECK-NEXT: orr.w r0, r0, r1, lsl #16
#[no_mangle]
pub extern "cmse-nonsecure-entry" fn cmse_ret_with_wide_u8(a: u8, b: u8) -> [WideU8; 2] {
    [WideU8 { a }, WideU8 { a: b }]
}

// CHECK-LABEL: cmse_ret_with_wide_u8_uninit:
// CHECK: mov r7, sp
// CHECK-NEXT: uxtb r0, r0
// CHECK-NEXT: orr.w r0, r0, r1, lsl #16
// CHECK-NEXT: bic r0, r0, #-16711936
#[no_mangle]
pub extern "cmse-nonsecure-entry" fn cmse_ret_with_wide_u8_uninit(
    a: u16,
    b: u16,
) -> [MaybeUninit<WideU8>; 2] {
    unsafe { [mem::transmute(a), mem::transmute(b)] }
}

// CHECK-LABEL: cmse_ret_with_wide_u8_uninit_tuple:
// CHECK: mov r7, sp
// CHECK-NEXT: uxtb r0, r0
// CHECK-NEXT: orr.w r0, r0, r1, lsl #16
// CHECK-NEXT: bic r0, r0, #-16711936
#[no_mangle]
pub extern "cmse-nonsecure-entry" fn cmse_ret_with_wide_u8_uninit_tuple(
    a: u16,
    b: u16,
) -> (MaybeUninit<WideU8>, MaybeUninit<WideU8>) {
    unsafe { (mem::transmute(a), mem::transmute(b)) }
}
