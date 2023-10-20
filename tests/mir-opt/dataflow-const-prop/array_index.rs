// skip-filecheck
// EMIT_MIR_FOR_EACH_PANIC_STRATEGY
// unit-test: DataflowConstProp
// EMIT_MIR_FOR_EACH_BIT_WIDTH

// EMIT_MIR array_index.main.DataflowConstProp.diff
fn main() {
    let x: u32 = [0, 1, 2, 3][2];
}
