//@ test-mir-pass: GVN
//@ compile-flags: -O --crate-type=lib
//@ ignore-endian-big
// EMIT_MIR_FOR_EACH_BIT_WIDTH

use std::mem::transmute;

// EMIT_MIR transmute.less_as_i8.GVN.diff
pub fn less_as_i8() -> i8 {
    // CHECK-LABEL: fn less_as_i8(
    // CHECK: _0 = const -1_i8;
    unsafe { transmute(std::cmp::Ordering::Less) }
}

// EMIT_MIR transmute.from_char.GVN.diff
pub fn from_char() -> i32 {
    // CHECK-LABEL: fn from_char(
    // CHECK: _0 = const 82_i32;
    unsafe { transmute('R') }
}

// EMIT_MIR transmute.valid_char.GVN.diff
pub fn valid_char() -> char {
    // CHECK-LABEL: fn valid_char(
    // CHECK: _0 = const 'R';
    unsafe { transmute(0x52_u32) }
}

// EMIT_MIR transmute.invalid_char.GVN.diff
pub unsafe fn invalid_char() -> char {
    // CHECK-LABEL: fn invalid_char(
    // CHECK: _0 = const {transmute(0x7fffffff): char};
    unsafe { transmute(i32::MAX) }
}

// EMIT_MIR transmute.invalid_bool.GVN.diff
pub unsafe fn invalid_bool() -> bool {
    // CHECK-LABEL: fn invalid_bool(
    // CHECK: _0 = const {transmute(0xff): bool};
    unsafe { transmute(-1_i8) }
}

// EMIT_MIR transmute.undef_union_as_integer.GVN.diff
pub unsafe fn undef_union_as_integer() -> u32 {
    // CHECK-LABEL: fn undef_union_as_integer(
    // CHECK: _0 = const <uninit>;
    union Union32 {
        value: u32,
        unit: (),
    }
    unsafe { transmute(Union32 { unit: () }) }
}

// EMIT_MIR transmute.unreachable_direct.GVN.diff
pub unsafe fn unreachable_direct() -> ! {
    // CHECK-LABEL: fn unreachable_direct(
    // CHECK: = const ();
    // CHECK: = const ZeroSized: Never;
    let x: Never = unsafe { transmute(()) };
    match x {}
}

enum Never {}
