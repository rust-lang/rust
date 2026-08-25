//@ test-mir-pass: DataflowConstProp
// EMIT_MIR_FOR_EACH_PANIC_STRATEGY
// EMIT_MIR_FOR_EACH_BIT_WIDTH

// EMIT_MIR large_array_index.main.DataflowConstProp.diff

// CHECK-LABEL: fn main(
fn main() {
    // check that we don't propagate this, because it's too large

    // CHECK: debug x => [[x:_.*]];
    // CHECK: [[array_lit:_.*]] = [const 0_u8; 5000];
    // CHECK: {{_.*}} = const 2_usize;
    // CHECK: {{_.*}} = const true;
    // CHECK: assert(const true
    // CHECK: [[x]] = copy [[array_lit]][2 of 3];
    let x: u8 = [0_u8; 5000][2];
}
