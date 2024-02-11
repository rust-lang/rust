// unit-test: GVN
// compile-flags: -Zmir-enable-passes=+InstSimplify
// EMIT_MIR_FOR_EACH_PANIC_STRATEGY
// EMIT_MIR_FOR_EACH_BIT_WIDTH

// EMIT_MIR slice_len.main.GVN.diff
fn main() {
    // CHECK-LABEL: fn main(
    // CHECK: debug a => [[a:_.*]];
    // CHECK: [[slice:_.*]] = const {{.*}} as &[u32] (PointerCoercion(Unsize));
    // CHECK: assert(const true,
    // CHECK: [[a]] = const 2_u32;
    let a = (&[1u32, 2, 3] as &[u32])[1];
}
