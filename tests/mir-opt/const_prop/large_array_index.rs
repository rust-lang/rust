//@ test-mir-pass: GVN
// EMIT_MIR_FOR_EACH_PANIC_STRATEGY
// EMIT_MIR_FOR_EACH_BIT_WIDTH

// EMIT_MIR large_array_index.main.GVN.diff
fn main() {
    // check that gvn access repeat inner.
    // CHECK-LABEL: fn main(
    // CHECK: debug x => [[x:_.*]];
    // CHECK: assert(const true,
    // CHECK: [[x]] = const 0_u8;
    let x: u8 = [0_u8; 5000][2];
}
