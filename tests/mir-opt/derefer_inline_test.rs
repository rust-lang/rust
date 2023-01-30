// unit-test: Derefer
// EMIT_MIR derefer_inline_test.main.Derefer.diff
// ignore-wasm32 compiled with panic=abort by default

#![feature(box_syntax)]
#[inline]
fn f() -> Box<u32> {
    box 0
}
fn main() {
    box f();
}
