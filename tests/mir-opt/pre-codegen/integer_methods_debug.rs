//@ compile-flags: -Copt-level=0 -Zmir-opt-level=1 -Cdebuginfo=limited
// EMIT_MIR_FOR_EACH_PANIC_STRATEGY

#![crate_type = "lib"]

// EMIT_MIR integer_methods_debug.cast_and_add.PreCodegen.after.mir
pub fn cast_and_add(x: i32) -> u32 {
    // CHECK-LABEL: fn cast_and_add(_1: i32) -> u32
    // CHECK: (inlined {{.+}}<impl i32>::cast_unsigned)
    // CHECK: (inlined {{.+}}<impl u32>::wrapping_add)
    // CHECK: _2 = copy _1 as u32 (IntToInt);
    // CHECK: _0 = Add(copy _2, const 42_u32);
    x.cast_unsigned().wrapping_add(42)
}
