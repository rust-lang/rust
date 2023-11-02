// EMIT_MIR_FOR_EACH_PANIC_STRATEGY
#![crate_type = "lib"]
#![feature(unchecked_shifts)]

// ignore-debug: the debug assertions prevent the inlining we are testing for
// compile-flags: -Zmir-opt-level=2 -Zinline-mir

// EMIT_MIR unchecked_shifts.unchecked_shl_unsigned_smaller.Inline.diff
// EMIT_MIR unchecked_shifts.unchecked_shl_unsigned_smaller.PreCodegen.after.mir
pub unsafe fn unchecked_shl_unsigned_smaller(a: u16, b: u32) -> u16 {
    // CHECK-LABEL: fn unchecked_shl_unsigned_smaller(
    // CHECK: (inlined core::num::<impl u16>::unchecked_shl)
    a.unchecked_shl(b)
}

// EMIT_MIR unchecked_shifts.unchecked_shr_signed_smaller.Inline.diff
// EMIT_MIR unchecked_shifts.unchecked_shr_signed_smaller.PreCodegen.after.mir
pub unsafe fn unchecked_shr_signed_smaller(a: i16, b: u32) -> i16 {
    // CHECK-LABEL: fn unchecked_shr_signed_smaller(
    // CHECK: (inlined core::num::<impl i16>::unchecked_shr)
    a.unchecked_shr(b)
}

// EMIT_MIR unchecked_shifts.unchecked_shl_unsigned_bigger.Inline.diff
// EMIT_MIR unchecked_shifts.unchecked_shl_unsigned_bigger.PreCodegen.after.mir
pub unsafe fn unchecked_shl_unsigned_bigger(a: u64, b: u32) -> u64 {
    // CHECK-LABEL: fn unchecked_shl_unsigned_bigger(
    // CHECK: (inlined core::num::<impl u64>::unchecked_shl)
    a.unchecked_shl(b)
}

// EMIT_MIR unchecked_shifts.unchecked_shr_signed_bigger.Inline.diff
// EMIT_MIR unchecked_shifts.unchecked_shr_signed_bigger.PreCodegen.after.mir
pub unsafe fn unchecked_shr_signed_bigger(a: i64, b: u32) -> i64 {
    // CHECK-LABEL: fn unchecked_shr_signed_bigger(
    // CHECK: (inlined core::num::<impl i64>::unchecked_shr)
    a.unchecked_shr(b)
}
