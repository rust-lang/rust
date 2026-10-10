#![allow(dead_code)]
#![allow(unused_imports)]

#[macro_use]
mod macros;

mod fs;
mod io;
mod miri_extern;

pub use self::fs::*;
pub use self::io::*;
pub use self::miri_extern::*;

pub fn run_provenance_gc() {
    // SAFETY: No preconditions. The GC is fine to run at any time.
    unsafe { miri_run_provenance_gc() }
}

/// Check that the function produces the intended set of outcomes.
#[track_caller]
pub fn check_all_outcomes<T: Eq + std::hash::Hash + std::fmt::Debug>(
    expected: impl IntoIterator<Item = T>,
    generate: impl Fn() -> T,
) {
    use std::collections::HashSet;

    let expected: HashSet<T> = HashSet::from_iter(expected);
    let mut seen = HashSet::new();
    // Let's give it N times as many tries as we are expecting values.
    let min_tries = std::cmp::max(20, expected.len() * 4);
    let max_tries = expected.len() * 50;
    for i in 0..max_tries {
        let val = generate();
        assert!(expected.contains(&val), "got an unexpected value: {val:?}");
        seen.insert(val);
        if i >= min_tries && expected.len() == seen.len() {
            // We saw everything and we did enough tries, let's avoid wasting time.
            return;
        }
    }
    // Let's see if we saw them all.
    if expected.len() == seen.len() {
        return;
    }
    // Find the missing one.
    for val in expected {
        if !seen.contains(&val) {
            panic!("did not get value that should be possible: {val:?}");
        }
    }
    unreachable!()
}

/// Check that the operation is non-deterministic
#[track_caller]
pub fn check_nondet<T: PartialEq + std::fmt::Debug>(f: impl Fn() -> T) {
    // We test some rather unlikely events with this, such as two global allocations getting the
    // same "salt" (1/32 chance). So give this *many* shots before we consider the test to have
    // failed.
    let rounds = 500;
    let first = f();
    for _ in 1..rounds {
        if f() != first {
            // We saw two different values!
            return;
        }
    }
    // We saw the same thing N times.
    panic!("expected non-determinism, got {rounds} times the same result: {first:?}");
}
