//! Regression test for issue #1399
//!
//! This tests that when a variable is used (via clone) and then later
//! captured by a closure, the last-use analysis doesn't incorrectly optimize
//! the earlier use as a "last use" and perform an invalid move.
//!
//! The sequence being tested:
//! 1. Create variable `k`
//! 2. Use `k.clone()` for some purpose
//! 3. Later capture `k` in a closure
//!
//! The analysis must not treat step 2 as the "last use" since step 3 needs `k`.
//!
//! See: https://github.com/rust-lang/rust/issues/1399

//@ run-pass

struct A {
    _a: Box<isize>,
}

pub fn main() {
    fn invoke<F>(f: F)
    where
        F: FnOnce(),
    {
        f();
    }

    let k: Box<_> = 22.into();

    // This clone should NOT be treated as "last use" of k
    // even though k is not used again until the closure
    let _u = A { _a: k.clone() };

    // Here k is actually captured by the closure
    // The last-use analyzer must have accounted for this when processing the clone above
    invoke(|| println!("{}", k.clone()));
}
