// unit-test: DataflowConstProp
// EMIT_MIR_FOR_EACH_PANIC_STRATEGY
// EMIT_MIR_FOR_EACH_BIT_WIDTH

// EMIT_MIR repeat.main.DataflowConstProp.diff
// CHECK-LABEL: fn main
fn main() {
    // CHECK: debug x => [[x:_.*]];

    // CHECK: {{_[0-9]+}} = const 8_usize;
    // CHECK: {{_[0-9]+}} = const true;
    // CHECK-LABEL: assert(const true

    // CHECK: {{_[0-9]+}} = {{_[0-9]+}}[2 of 3];
    // CHECK: [[x]] = Add(move {{_[0-9]+}}, const 0_u32);
    let x: u32 = [42; 8][2] + 0;
}
