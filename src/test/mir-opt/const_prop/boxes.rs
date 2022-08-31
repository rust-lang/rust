// unit-test: ConstProp
// compile-flags: -O
// ignore-emscripten compiled with panic=abort by default
// ignore-wasm32
// ignore-wasm64

#![feature(box_syntax)]

// Note: this test verifies that we, in fact, do not const prop `box`

// EMIT_MIR boxes.main.ConstProp.diff
fn main() {
    let x = *(box 42) + 0;
}
