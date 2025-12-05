//! Test that opt-level=z produces correct code on Windows MSVC targets.
//!
//! A previously outdated version of LLVM caused compilation failures and
//! generated invalid code on Windows specifically with optimization level `z`.
//! The bug manifested as corrupted base pointers due to incorrect register
//! usage in the generated assembly (e.g., `popl %esi` corrupting local variables).
//! After updating to a more recent LLVM version, this test ensures that
//! compilation and execution both succeed with opt-level=z.
//!
//! Regression test for <https://github.com/rust-lang/rust/issues/45034>.

//@ ignore-cross-compile
// Reason: the compiled binary is executed
//@ only-windows
// Reason: the observed bug only occurred on Windows MSVC targets
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
    // This call used to trigger an LLVM bug in opt-level=z where the base
    // pointer gets corrupted due to incorrect register allocation
    let y: fn(i32, i32) -> i64 = test::black_box(foo);
    test::black_box(y(1, 2));
}

fn main() {
    bar();
}
