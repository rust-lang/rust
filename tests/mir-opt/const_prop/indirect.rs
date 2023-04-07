// ignore-wasm32 compiled with panic=abort by default
// unit-test: ConstProp
// compile-flags: -C overflow-checks=on

// EMIT_MIR indirect.main.ConstProp.diff
fn main() {
    let x = (2u32 as u8) + 1;
}
