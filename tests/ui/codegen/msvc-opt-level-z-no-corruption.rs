// A previously outdated version of LLVM caused compilation failures on Windows
// specifically with optimization level `z`. After the update to a more recent LLVM
// version, this test checks that compilation and execution both succeed.
// See https://github.com/rust-lang/rust/issues/45034

//@ ignore-cross-compile
// Reason: the compiled binary is executed
//@ only-windows
// Reason: the observed bug only occurs on Windows
//@ run-pass
//@ compile-flags: -C opt-level=z

#![feature(test)]
extern crate test;

fn foo(x: i32, y: i32) -> i64 {
    (x + y) as i64
}

#[inline(never)]
fn bar() {
    let _f = Box::new(0);
    // This call used to trigger an LLVM bug in opt-level z where the base
    // pointer gets corrupted, see issue #45034
    let y: fn(i32, i32) -> i64 = test::black_box(foo);
    test::black_box(y(1, 2));
}

fn main() {
    bar();
}
