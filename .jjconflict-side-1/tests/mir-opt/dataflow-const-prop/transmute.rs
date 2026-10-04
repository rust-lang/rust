//@ test-mir-pass: DataflowConstProp
//@ compile-flags: -O --crate-type=lib
//@ ignore-endian-big
// EMIT_MIR_FOR_EACH_BIT_WIDTH

use std::mem::transmute;

// EMIT_MIR transmute.less_as_i8.DataflowConstProp.diff
pub fn less_as_i8() -> i8 {
    // CHECK-LABEL: fn less_as_i8(
    // FIXME-CHECK: _0 = const -1_i8;
    unsafe { transmute(std::cmp::Ordering::Less) }
}

// EMIT_MIR transmute.from_char.DataflowConstProp.diff
pub fn from_char() -> i32 {
    // CHECK-LABEL: fn from_char(
    // CHECK: _0 = const 82_i32;
    unsafe { transmute('R') }
}

// EMIT_MIR transmute.valid_char.DataflowConstProp.diff
pub fn valid_char() -> char {
    // CHECK-LABEL: fn valid_char(
    // CHECK: _0 = const 'R';
    unsafe { transmute(0x52_u32) }
}

// EMIT_MIR transmute.invalid_char.DataflowConstProp.diff
pub unsafe fn invalid_char() -> char {
    // CHECK-LABEL: fn invalid_char(
    // CHECK: _0 = const {transmute(0x7fffffff): char};
    unsafe { transmute(i32::MAX) }
}

// EMIT_MIR transmute.invalid_bool.DataflowConstProp.diff
pub unsafe fn invalid_bool() -> bool {
    // CHECK-LABEL: fn invalid_bool(
    // CHECK: _0 = const {transmute(0xff): bool};
    unsafe { transmute(-1_i8) }
}

// EMIT_MIR transmute.undef_union_as_integer.DataflowConstProp.diff
pub unsafe fn undef_union_as_integer() -> u32 {
    // CHECK-LABEL: fn undef_union_as_integer(
    // CHECK: _1 = Union32 {
    // CHECK: _0 = move _1 as u32 (Transmute);
    union Union32 {
        value: u32,
        unit: (),
    }
    unsafe { transmute(Union32 { unit: () }) }
}

// EMIT_MIR transmute.unreachable_direct.DataflowConstProp.diff
pub unsafe fn unreachable_direct() -> ! {
    // CHECK-LABEL: fn unreachable_direct(
    // CHECK: = const ();
    // CHECK: = const ZeroSized: Never;
    let x: Never = unsafe { transmute(()) };
    match x {}
}

// EMIT_MIR transmute.unreachable_ref.DataflowConstProp.diff
pub unsafe fn unreachable_ref() -> ! {
    // CHECK-LABEL: fn unreachable_ref(
    // CHECK: = const {0x1 as &Never};
    let x: &Never = unsafe { transmute(1_usize) };
    match *x {}
}

// EMIT_MIR transmute.unreachable_mut.DataflowConstProp.diff
pub unsafe fn unreachable_mut() -> ! {
    // CHECK-LABEL: fn unreachable_mut(
    // CHECK: = const {0x1 as &mut Never};
    let x: &mut Never = unsafe { transmute(1_usize) };
    match *x {}
}

// EMIT_MIR transmute.unreachable_box.DataflowConstProp.diff
pub unsafe fn unreachable_box() -> ! {
    // CHECK-LABEL: fn unreachable_box(
    // CHECK: = const Box::<Never>(
    let x: Box<Never> = unsafe { transmute(1_usize) };
    match *x {}
}

enum Never {}
