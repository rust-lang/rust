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
//
// When passing arguments the current implementation of the C ABI already clears padding sometimes,
// but it's not something that can be relied on.

extern crate minicore;
use minicore::*;

#[repr(C)]
struct InnerPadding {
    a: u8,
    b: u16,
}

// CHECK-LABEL: c_ret_with_inner_padding:
// CHECK: mov r7, sp
// CHECK-NEXT: orr.w r0, r0, r1, lsl #16
#[no_mangle]
extern "C" fn c_ret_with_inner_padding(a: u8, b: u16) -> InnerPadding {
    InnerPadding { a, b }
}

// CHECK-LABEL: cmse_ret_with_inner_padding:
// CHECK: mov r7, sp
// CHECK-NEXT: uxtb r0, r0
// CHECK-NEXT: orr.w r0, r0, r1, lsl #16
#[no_mangle]
extern "cmse-nonsecure-entry" fn cmse_ret_with_inner_padding(a: u8, b: u16) -> InnerPadding {
    InnerPadding { a, b }
}

// CHECK-LABEL: c_call_with_inner_padding:
// CHECK: mov r7, sp
// CHECK-NEXT: mov r2, r0
// CHECK-NEXT: bic r0, r1, #65280
// CHECK-NEXT: pop.w   {r7, lr}
#[no_mangle]
extern "C" fn c_call_with_inner_padding(f: unsafe extern "C" fn(InnerPadding), x: InnerPadding) {
    unsafe { f(x) }
}

// CHECK-LABEL: cmse_call_with_inner_padding:
// CHECK: mov r7, sp
// CHECK-NEXT: mov r2, r0
// CHECK-NEXT: bic r0, r1, #65280
// CHECK-NEXT: push.w  {r4, r5, r6, r7, r8, r9, r10, r11}
#[no_mangle]
extern "C" fn cmse_call_with_inner_padding(
    f: unsafe extern "cmse-nonsecure-call" fn(InnerPadding),
    x: InnerPadding,
) {
    unsafe { f(x) }
}

#[repr(C)]
struct TrailingPadding {
    a: u16,
    b: u8,
}

// CHECK-LABEL: c_ret_with_trailing_padding:
// CHECK: mov r7, sp
// CHECK-NEXT: orr.w r0, r0, r1, lsl #16
#[no_mangle]
extern "C" fn c_ret_with_trailing_padding(a: u16, b: u8) -> TrailingPadding {
    TrailingPadding { a, b }
}

// CHECK-LABEL: cmse_ret_with_trailing_padding:
// CHECK: mov r7, sp
// CHECK-NEXT: uxtb r1, r1
// CHECK-NEXT: uxth r0, r0
// CHECK-NEXT: orr.w r0, r0, r1, lsl #16
#[no_mangle]
extern "cmse-nonsecure-entry" fn cmse_ret_with_trailing_padding(a: u16, b: u8) -> TrailingPadding {
    TrailingPadding { a, b }
}

// CHECK-LABEL: c_call_with_trailing_padding:
// CHECK: mov r7, sp
// CHECK-NEXT: mov r2, r0
// CHECK-NEXT: bic r0, r1, #-16777216
// CHECK-NEXT: pop.w   {r7, lr}
#[no_mangle]
extern "C" fn c_call_with_trailing_padding(
    f: unsafe extern "C" fn(TrailingPadding),
    x: TrailingPadding,
) {
    unsafe { f(x) }
}

// CHECK-LABEL: cmse_call_with_trailing_padding:
// CHECK: mov r7, sp
// CHECK-NEXT: mov r2, r0
// CHECK-NEXT: bic r0, r1, #-16777216
// CHECK-NEXT: push.w {r4, r5, r6, r7, r8, r9, r10, r11}
#[no_mangle]
extern "C" fn cmse_call_with_trailing_padding(
    f: unsafe extern "cmse-nonsecure-call" fn(TrailingPadding),
    x: TrailingPadding,
) {
    unsafe { f(x) }
}

#[repr(C, align(2))]
struct WideU8 {
    a: u8,
}

// `extern "C"` does not clear the padding.
//
// CHECK-LABEL: c_ret_with_wide_u8:
// CHECK: mov r7, sp
// CHECK-NEXT: orr.w r0, r0, r1, lsl #16
#[no_mangle]
extern "C" fn c_ret_with_wide_u8(a: u8, b: u8) -> [WideU8; 2] {
    [WideU8 { a }, WideU8 { a: b }]
}

// Upper bits are cleared by uxtb.
//
// CHECK-LABEL: cmse_ret_with_wide_u8:
// CHECK: mov r7, sp
// CHECK-NEXT: uxtb r1, r1
// CHECK-NEXT: uxtb r0, r0
// CHECK-NEXT: orr.w r0, r0, r1, lsl #16
#[no_mangle]
extern "cmse-nonsecure-entry" fn cmse_ret_with_wide_u8(a: u8, b: u8) -> [WideU8; 2] {
    [WideU8 { a }, WideU8 { a: b }]
}

// Same idea, the padding is recognized even through the MaybeUninit.
//
// CHECK-LABEL: cmse_ret_with_wide_u8_uninit:
// CHECK: mov r7, sp
// CHECK-NEXT: uxtb r0, r0
// CHECK-NEXT: orr.w r0, r0, r1, lsl #16
// CHECK-NEXT: bic r0, r0, #-16711936
#[no_mangle]
extern "cmse-nonsecure-entry" fn cmse_ret_with_wide_u8_uninit(
    a: u16,
    b: u16,
) -> [MaybeUninit<WideU8>; 2] {
    unsafe { [mem::transmute(a), mem::transmute(b)] }
}

// Same idea, the padding is recognized even through the MaybeUninit.
//
// CHECK-LABEL: cmse_ret_with_wide_u8_uninit_tuple:
// CHECK: mov r7, sp
// CHECK-NEXT: uxtb r0, r0
// CHECK-NEXT: orr.w r0, r0, r1, lsl #16
// CHECK-NEXT: bic r0, r0, #-16711936
#[no_mangle]
extern "cmse-nonsecure-entry" fn cmse_ret_with_wide_u8_uninit_tuple(
    a: u16,
    b: u16,
) -> (MaybeUninit<WideU8>, MaybeUninit<WideU8>) {
    unsafe { (mem::transmute(a), mem::transmute(b)) }
}

// CHECK-LABEL: c_call_with_inner_wide_u8:
// CHECK: push    {r7, lr}
// CHECK-NEXT: .setfp  r7, sp
// CHECK-NEXT: mov r7, sp
// CHECK-NEXT: mov lr, r3
// CHECK-NEXT: mov r12, r0
// CHECK-NEXT: ldr r3, [r7, #8]
// CHECK-NEXT: mov r0, r1
// CHECK-NEXT: mov r1, r2
// CHECK-NEXT: mov r2, lr
// CHECK-NEXT: pop.w   {r7, lr}
// CHECK-NEXT: bx  r12
#[no_mangle]
extern "C" fn c_call_with_inner_wide_u8(f: unsafe extern "C" fn([WideU8; 8]), x: [WideU8; 8]) {
    unsafe { f(x) }
}

// CHECK-LABEL: cmse_call_with_inner_wide_u8:
// CHECK: push    {r7, lr}
// CHECK-NEXT: .setfp  r7, sp
// CHECK-NEXT: mov r7, sp
// CHECK-NEXT: mov r12, r0
// CHECK-NEXT: bic r0, r1, #-16711936
// CHECK-NEXT: bic r1, r2, #-16711936
// CHECK-NEXT: bic r2, r3, #-16711936
// CHECK-NEXT: ldr r3, [r7, #8]
// CHECK-NEXT: bic r3, r3, #-16711936
// CHECK-NEXT: push.w  {r4, r5, r6, r7, r8, r9, r10, r11}
#[no_mangle]
extern "C" fn cmse_call_with_inner_wide_u8(
    f: unsafe extern "cmse-nonsecure-call" fn([WideU8; 8]),
    x: [WideU8; 8],
) {
    unsafe { f(x) }
}

/// No variant-dependent padding.
#[repr(C)]
enum VariantsSameSize {
    A(u16),
    B(u16),
}
impl Copy for VariantsSameSize {}

// CHECK-LABEL: variants_same_size:
// CHECK: mov r7, sp
// CHECK-NEXT: ldrh r0, [r0, #2]
// CHECK-NEXT: lsls r0, r0, #16
#[no_mangle]
extern "cmse-nonsecure-entry" fn variants_same_size(v: &VariantsSameSize) -> VariantsSameSize {
    *v
}

/// One byte of variant-dependent padding.
#[repr(C)]
enum VariantsDifferentSize {
    A(u8),
    B(u16),
}
impl Copy for VariantsDifferentSize {}

// CHECK-LABEL: variants_different_size:
// CHECK: mov r7, sp
// CHECK-NEXT: ldrh r0, [r0, #2]
// CHECK-NEXT: lsls r0, r0, #16
#[no_mangle]
extern "cmse-nonsecure-entry" fn variants_different_size(
    v: &VariantsDifferentSize,
) -> VariantsDifferentSize {
    *v
}

enum Void {}
impl Copy for Void {}

#[repr(C)]
enum UninhabitedVariant {
    A(Void),
    B(u16),
}
impl Copy for UninhabitedVariant {}

// CHECK-LABEL: uninhabited_variant:
// CHECK: mov r7, sp
// CHECK-NEXT: ldrh r0, [r0, #2]
// CHECK-NEXT: lsls r0, r0, #16
#[no_mangle]
extern "cmse-nonsecure-entry" fn uninhabited_variant(v: &UninhabitedVariant) -> UninhabitedVariant {
    *v
}

// CHECK-LABEL: variants_same_size_array:
// CHECK: mov r7, sp
// CHECK-NEXT: ldrh r0, [r0, #2]
// CHECK-NEXT: lsls r0, r0, #16
#[no_mangle]
#[expect(improper_ctypes_definitions)]
extern "cmse-nonsecure-entry" fn variants_same_size_array(
    v: &[VariantsSameSize; 1],
) -> [VariantsSameSize; 1] {
    *v
}

// CHECK-LABEL: variants_different_size_array:
// CHECK: mov r7, sp
// CHECK-NEXT: ldrh r0, [r0, #2]
// CHECK-NEXT: lsls r0, r0, #16
#[no_mangle]
#[expect(improper_ctypes_definitions)]
extern "cmse-nonsecure-entry" fn variants_different_size_array(
    v: &[VariantsDifferentSize; 1],
) -> [VariantsDifferentSize; 1] {
    *v
}

// CHECK-LABEL: variants_same_size_tuple:
// CHECK: mov r7, sp
// CHECK-NEXT: ldrh r0, [r0, #2]
// CHECK-NEXT: lsls r0, r0, #16
#[no_mangle]
#[expect(improper_ctypes_definitions)]
extern "cmse-nonsecure-entry" fn variants_same_size_tuple(
    v: &(VariantsSameSize,),
) -> (VariantsSameSize,) {
    *v
}

// CHECK-LABEL: variants_different_size_tuple:
// CHECK: mov r7, sp
// CHECK-NEXT: ldrh r0, [r0, #2]
// CHECK-NEXT: lsls r0, r0, #16
#[no_mangle]
#[expect(improper_ctypes_definitions)]
extern "cmse-nonsecure-entry" fn variants_different_size_tuple(
    v: &(VariantsDifferentSize,),
) -> (VariantsDifferentSize,) {
    *v
}

/// Three variants of different sizes.
#[repr(C)]
enum ThreeVariants {
    A(u8),
    B(u16),
    C(u32),
}

// CHECK-LABEL: cmse_call_three_variants:
// CHECK: mov r7, sp
// CHECK-NEXT: mov r1, r2
// CHECK-NEXT: mov r2, r0
// CHECK-NEXT: movs r0, #0
#[no_mangle]
#[expect(improper_ctypes_definitions)]
extern "C" fn cmse_call_three_variants(
    f: unsafe extern "cmse-nonsecure-call" fn(ThreeVariants),
    x: ThreeVariants,
) {
    unsafe { f(x) }
}

/// The tag is stored in the niche of the `bool`.
#[repr(C)]
struct BoolU32 {
    flag: bool,
    val: u32,
}

// CHECK-LABEL: cmse_call_niche:
// CHECK: mov r7, sp
// CHECK-NEXT: mov r3, r0
// CHECK-NEXT: uxtb r0, r1
// CHECK-NEXT: mov r1, r2
#[no_mangle]
#[expect(improper_ctypes_definitions)]
extern "C" fn cmse_call_niche(
    f: unsafe extern "cmse-nonsecure-call" fn(Option<BoolU32>),
    x: Option<BoolU32>,
) {
    unsafe { f(x) }
}
