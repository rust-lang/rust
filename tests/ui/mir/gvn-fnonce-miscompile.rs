//! Regression test for a GVN miscompile that turned two `move`d closures into a double-use of
//! the same stack slot, because GVN unified the value-numbers of two distinct empty `Vec`-tuple
//! aggregates and rewrote the later one into `Operand::Copy(earlier_local)` after that earlier
//! local had already been moved out of by an `FnOnce` call.
//!
//! See <https://github.com/rust-lang/rust/issues/155241>.
//!
//! Without the fix, the second `with(...)` call observes the first closure's `Vec` (or freed
//! memory) instead of a freshly-constructed empty `Vec`, leading to either an `assert_eq!`
//! failure or heap corruption (SIGABRT) at `-Copt-level=2` and above.

//@ run-pass
//@ compile-flags: -Copt-level=3 -Zmir-enable-passes=+GVN

fn with(f: impl FnOnce(Vec<usize>)) {
    f(Vec::new())
}

fn main() {
    with(|mut v| v.resize(2, 1));
    with(|v| assert_eq!(v.len(), 0, "second closure must observe a fresh, empty Vec"));
}
