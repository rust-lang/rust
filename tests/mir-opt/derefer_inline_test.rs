// skip-filecheck
//@ test-mir-pass: Derefer
// EMIT_MIR derefer_inline_test.main.Derefer.diff
// EMIT_MIR_FOR_EACH_PANIC_STRATEGY

#[inline]
fn f() -> Box<u32> {
    Box::new(0)
}
fn main() {
    Box::new(f());
}
