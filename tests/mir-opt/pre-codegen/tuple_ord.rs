//@ compile-flags: -O -Zmir-opt-level=2 -Cdebuginfo=0
//@ needs-unwind

#![crate_type = "lib"]

// EMIT_MIR tuple_ord.demo_le_total.PreCodegen.after.mir
pub fn demo_le_total(a: &(u16, i16), b: &(u16, i16)) -> bool {
    // CHECK-LABEL: demo_le_total
    a <= b
}

// EMIT_MIR tuple_ord.demo_ge_partial.PreCodegen.after.mir
pub fn demo_ge_partial(a: &(f32, f32), b: &(f32, f32)) -> bool {
    // CHECK-LABEL: demo_ge_partial
    a >= b
}
