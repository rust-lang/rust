// EMIT_MIR_FOR_EACH_PANIC_STRATEGY
// unit-test: DataflowConstProp
// compile-flags: -Zmir-enable-passes=+InstSimplify
// EMIT_MIR_FOR_EACH_BIT_WIDTH

// EMIT_MIR slice_len.main.DataflowConstProp.diff

// CHECK-LABEL: fn main
fn main() {
    // CHECK: debug local => [[local:_[0-9]+]];
    // CHECK: debug constant => [[constant:_[0-9]+]];

    // CHECK: {{_[0-9]+}} = const 3_usize;
    // CHECK: {{_[0-9]+}} = const true;

    // CHECK: [[local]] = (*{{_[0-9]+}})[1 of 2];
    let local = (&[1u32, 2, 3] as &[u32])[1];

    const SLICE: &[u32] = &[1, 2, 3];

    // CHECK: [[constant]] = (*{{_[0-9]+}})[1 of 2];
    let constant = SLICE[1];
}
