// unit-test: ConstProp
// ignore-wasm32 compiled with panic=abort by default
// EMIT_MIR scalar_literal_propagation.main.ConstProp.diff
fn main() {
    let x = 1;
    consume(x);
}

#[inline(never)]
fn consume(_: u32) { }
