// EMIT_MIR_FOR_EACH_PANIC_STRATEGY
#![crate_type = "lib"]
#![feature(unchecked_shifts)]

//@ compile-flags: -Zmir-opt-level=2 -Zinline-mir

// These used to be more interesting when the library had to fix the RHS type.
// After MCP#693, though, that's the backend's problem, not something in MIR.

// EMIT_MIR unchecked_shifts.unchecked_shl_unsigned_smaller.Inline.diff
// EMIT_MIR unchecked_shifts.unchecked_shl_unsigned_smaller.PreCodegen.after.mir
pub unsafe fn unchecked_shl_unsigned_smaller(a: u16, b: u32) -> u16 {
    // CHECK-LABEL: fn unchecked_shl_unsigned_smaller(
    // CHECK: (inlined #[track_caller] core::num::<impl u16>::unchecked_shl)
    a.unchecked_shl(b)
}

// EMIT_MIR unchecked_shifts.unchecked_shr_signed_bigger.Inline.diff
// EMIT_MIR unchecked_shifts.unchecked_shr_signed_bigger.PreCodegen.after.mir
pub unsafe fn unchecked_shr_signed_bigger(a: i64, b: u32) -> i64 {
    // CHECK-LABEL: fn unchecked_shr_signed_bigger(
    // CHECK: (inlined #[track_caller] core::num::<impl i64>::unchecked_shr)
    a.unchecked_shr(b)
}
