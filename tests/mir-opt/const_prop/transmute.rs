// unit-test: ConstProp
// compile-flags: -O --crate-type=lib

use std::mem::transmute;

// EMIT_MIR transmute.less_as_i8.ConstProp.diff
pub fn less_as_i8() -> i8 {
    unsafe { transmute(std::cmp::Ordering::Less) }
}

// EMIT_MIR transmute.from_char.ConstProp.diff
pub fn from_char() -> i32 {
    unsafe { transmute('R') }
}

// EMIT_MIR transmute.valid_char.ConstProp.diff
pub fn valid_char() -> char {
    unsafe { transmute(0x52_u32) }
}

// EMIT_MIR transmute.invalid_char.ConstProp.diff
pub unsafe fn invalid_char() -> char {
    unsafe { transmute(i32::MAX) }
}

// EMIT_MIR transmute.invalid_bool.ConstProp.diff
pub unsafe fn invalid_bool() -> bool {
    unsafe { transmute(-1_i8) }
}

// EMIT_MIR transmute.undef_union_as_integer.ConstProp.diff
pub unsafe fn undef_union_as_integer() -> u32 {
    union Union32 { value: u32, unit: () }
    unsafe { transmute(Union32 { unit: () }) }
}

// EMIT_MIR transmute.unreachable_direct.ConstProp.diff
pub unsafe fn unreachable_direct() -> ! {
    let x: Never = unsafe { transmute(()) };
    match x {}
}

// EMIT_MIR transmute.unreachable_ref.ConstProp.diff
pub unsafe fn unreachable_ref() -> ! {
    let x: &Never = unsafe { transmute(1_usize) };
    match *x {}
}

// EMIT_MIR transmute.unreachable_mut.ConstProp.diff
pub unsafe fn unreachable_mut() -> ! {
    let x: &mut Never = unsafe { transmute(1_usize) };
    match *x {}
}

// EMIT_MIR transmute.unreachable_box.ConstProp.diff
pub unsafe fn unreachable_box() -> ! {
    let x: Box<Never> = unsafe { transmute(1_usize) };
    match *x {}
}

enum Never {}
