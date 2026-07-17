//! Regression test for <https://github.com/rust-lang/rust/issues/4759>.
//! This used to trigger LLVM assertion.
//@ run-pass

trait U { fn f(self); }
impl U for isize { fn f(self) {} }
pub fn main() { 4.f(); }
