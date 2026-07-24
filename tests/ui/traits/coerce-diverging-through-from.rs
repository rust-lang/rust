//! Regression test for <https://github.com/rust-lang/rust/issues/43853>.
//! Test using `From::from(f())` on a diverging `f()` doesn't ICE.
//@ run-pass
//@ needs-unwind

use std::panic;

fn test() {
    wait(|| panic!());
}

fn wait<T, F: FnOnce() -> T>(f: F) -> F::Output {
    From::from(f())
}

fn main() {
    let result = panic::catch_unwind(move || test());
    assert!(result.is_err());
}
