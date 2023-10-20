// skip-filecheck
// unit-test: ConstProp
// EMIT_MIR_FOR_EACH_PANIC_STRATEGY
// EMIT_MIR_FOR_EACH_BIT_WIDTH

// EMIT_MIR repeat.main.ConstProp.diff
fn main() {
    let x: u32 = [42; 8][2] + 0;
}
