// skip-filecheck
// EMIT_MIR_FOR_EACH_PANIC_STRATEGY
// unit-test: DataflowConstProp
// compile-flags: -Zmir-enable-passes=+InstSimplify
// EMIT_MIR_FOR_EACH_BIT_WIDTH

// EMIT_MIR slice_len.main.DataflowConstProp.diff
fn main() {
    let local = (&[1u32, 2, 3] as &[u32])[1];

    const SLICE: &[u32] = &[1, 2, 3];
    let constant = SLICE[1];
}
