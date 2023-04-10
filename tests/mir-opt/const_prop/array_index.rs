// ignore-wasm32 compiled with panic=abort by default
// unit-test: ConstProp
// EMIT_MIR_FOR_EACH_BIT_WIDTH

// EMIT_MIR array_index.main.ConstProp.diff
fn main() {
    let x: u32 = [0, 1, 2, 3][2];
}
