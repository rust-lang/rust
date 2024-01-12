// EMIT_MIR_FOR_EACH_PANIC_STRATEGY
// unit-test: DataflowConstProp
// EMIT_MIR_FOR_EACH_BIT_WIDTH

// EMIT_MIR array_index.main.DataflowConstProp.diff

// CHECK-LABEL: fn main() -> () {
fn main() {
    // CHECK: let mut [[array_lit:_.*]]: [u32; 4];
    // CHECK:     debug x => [[x:_.*]];

    let x: u32 = [0, 1, 2, 3][2];
    // CHECK:       [[array_lit]] = [const 0_u32, const 1_u32, const 2_u32, const 3_u32];
    // CHECK-LABEL: assert(const true,
    // CHECK:     [[x]] = [[array_lit]][2 of 3];
}
