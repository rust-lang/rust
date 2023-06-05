// unit-test: ConstProp
// ignore-wasm32 compiled with panic=abort by default
// compile-flags: -Zmir-enable-passes=+NormalizeArrayLen
// EMIT_MIR_FOR_EACH_BIT_WIDTH

// EMIT_MIR repeat.main.ConstProp.diff
fn main() {
    let x: u32 = [42; 8][2] + 0;
}
