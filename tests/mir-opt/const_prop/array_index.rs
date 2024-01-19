// unit-test: GVN
// EMIT_MIR_FOR_EACH_PANIC_STRATEGY
// EMIT_MIR_FOR_EACH_BIT_WIDTH

// EMIT_MIR array_index.main.GVN.diff
fn main() {
    // CHECK-LABEL: fn main(
    // CHECK: debug x => const 2_u32;
    let x: u32 = [0, 1, 2, 3][2];
}
