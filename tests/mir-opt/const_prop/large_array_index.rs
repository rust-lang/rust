// unit-test: ConstProp
// ignore-wasm32 compiled with panic=abort by default
// compile-flags: -Zmir-enable-passes=+NormalizeArrayLen
// EMIT_MIR_FOR_EACH_BIT_WIDTH

// EMIT_MIR large_array_index.main.ConstProp.diff
fn main() {
    // check that we don't propagate this, because it's too large
    let x: u8 = [0_u8; 5000][2];
}
