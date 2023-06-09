// unit-test: ConstProp
// compile-flags: -O
// ignore-emscripten compiled with panic=abort by default
// ignore-wasm32
// ignore-wasm64

#![feature(rustc_attrs, stmt_expr_attributes)]

// Note: this test verifies that we, in fact, do not const prop `#[rustc_box]`

// EMIT_MIR boxes.main.ConstProp.diff
fn main() {
    let x = *(#[rustc_box]
    Box::new(42))
        + 0;
}
