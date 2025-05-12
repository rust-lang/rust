//@ run-pass
//! Tests that the addition of guard patterns does not change the behavior of the `pat` macro
//! fragment.
#![feature(guard_patterns)]
#![allow(incomplete_features)]

macro_rules! has_guard {
    ($p:pat) => {
        false
    };
    ($p:pat if $e:expr) => {
        true
    };
}

fn main() {
    assert_eq!(has_guard!(Some(_)), false);
    assert_eq!(has_guard!(Some(_) if true), true);
    assert_eq!(has_guard!((Some(_) if true)), false);
}
