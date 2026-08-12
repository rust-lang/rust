//@ compile-flags: -O -Zmir-opt-level=2
// EMIT_MIR_FOR_EACH_PANIC_STRATEGY

#![crate_type = "lib"]

// EMIT_MIR array_ord.lt_ipv4.runtime-optimized.after.mir
pub unsafe fn lt_ipv4<T: Copy>(a: &[u8; 4], b: &[u8; 4]) -> bool {
    // CHECK-LABEL: fn lt_ipv4(_1: &[u8; 4], _2: &[u8; 4]) -> bool
    // CHECK: [[A1:_.+]] = &raw const (*_1);
    // CHECK: [[A2:_.+]] = copy [[A1]] as *const u8 (PtrToPtr);
    // CHECK: [[B1:_.+]] = &raw const (*_2);
    // CHECK: [[B2:_.+]] = copy [[B1]] as *const u8 (PtrToPtr);
    // CHECK: [[C1:_.+]] = compare_bytes(move [[A2]], move [[B2]], const 4_usize)
    // CHECK: [[C2:_.+]] = Cmp(copy [[C1]], const 0_i32);
    // CHECK: [[D:_.+]] = discriminant([[C2]]);
    // CHECK: _0 = Lt(move [[D]], const 0_i8);
    a < b
}
