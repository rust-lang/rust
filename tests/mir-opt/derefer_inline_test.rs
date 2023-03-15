// unit-test: Derefer
// EMIT_MIR derefer_inline_test.main.Derefer.diff
// ignore-wasm32 compiled with panic=abort by default

#[inline]
fn f() -> Box<u32> {
    Box::new(0)
}
fn main() {
    Box::new(f());
}
