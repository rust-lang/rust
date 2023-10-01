// unit-test: DataflowConstProp
// compile-flags: -O --crate-type=lib
// ignore-endian-big
// EMIT_MIR_FOR_EACH_BIT_WIDTH

use std::mem::transmute;

// EMIT_MIR transmute.less_as_i8.DataflowConstProp.diff
pub fn less_as_i8() -> i8 {
    unsafe { transmute(std::cmp::Ordering::Less) }
}

// EMIT_MIR transmute.from_char.DataflowConstProp.diff
pub fn from_char() -> i32 {
    unsafe { transmute('R') }
}

// EMIT_MIR transmute.valid_char.DataflowConstProp.diff
pub fn valid_char() -> char {
    unsafe { transmute(0x52_u32) }
}

// EMIT_MIR transmute.invalid_char.DataflowConstProp.diff
pub unsafe fn invalid_char() -> char {
    unsafe { transmute(i32::MAX) }
}

// EMIT_MIR transmute.invalid_bool.DataflowConstProp.diff
pub unsafe fn invalid_bool() -> bool {
    unsafe { transmute(-1_i8) }
}

// EMIT_MIR transmute.undef_union_as_integer.DataflowConstProp.diff
pub unsafe fn undef_union_as_integer() -> u32 {
    union Union32 { value: u32, unit: () }
    unsafe { transmute(Union32 { unit: () }) }
}

// EMIT_MIR transmute.unreachable_direct.DataflowConstProp.diff
pub unsafe fn unreachable_direct() -> ! {
    let x: Never = unsafe { transmute(()) };
    match x {}
}

// EMIT_MIR transmute.unreachable_ref.DataflowConstProp.diff
pub unsafe fn unreachable_ref() -> ! {
    let x: &Never = unsafe { transmute(1_usize) };
    match *x {}
}

// EMIT_MIR transmute.unreachable_mut.DataflowConstProp.diff
pub unsafe fn unreachable_mut() -> ! {
    let x: &mut Never = unsafe { transmute(1_usize) };
    match *x {}
}

// EMIT_MIR transmute.unreachable_box.DataflowConstProp.diff
pub unsafe fn unreachable_box() -> ! {
    let x: Box<Never> = unsafe { transmute(1_usize) };
    match *x {}
}

enum Never {}
