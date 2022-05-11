// EMIT_MIR derefer_inline_test.main.Derefer.diff
#![feature(box_syntax)]
#[inline]
fn f() -> Box<i32> {
    box 0
}
fn main() {
    box f();
}
