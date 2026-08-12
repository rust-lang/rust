//@ compile-flags: -O -Zmir-opt-level=2
// EMIT_MIR_FOR_EACH_PANIC_STRATEGY

#![crate_type = "lib"]

// EMIT_MIR array_ord.lt_ipv4.runtime-optimized.after.mir
pub unsafe fn lt_ipv4<T: Copy>(a: &[u8; 4], b: &[u8; 4]) -> bool {
    // CHECK-LABEL: fn lt_ipv4(_1: &[u8; 4], _2: &[u8; 4]) -> bool
    // CHECK: [[A:_.+]] = copy _1 as &[u8] (PointerCoercion(Unsize, Implicit));
    // CHECK: [[B:_.+]] = copy _2 as &[u8] (PointerCoercion(Unsize, Implicit));
    // CHECK: [[C:_.+]] = <u8 as core::slice::cmp::SliceOrd>::compare(move [[A]], move [[B]])
    // CHECK: [[D:_.+]] = discriminant(_5);
    // CHECK: _0 = Lt(move [[D]], const 0_i8);
    a < b
}
