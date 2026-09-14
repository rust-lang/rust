//! Regression test for <https://github.com/rust-lang/rust/issues/44216>.
//! Test overflowing `Instant` panics.
//@ run-fail
//@ error-pattern:overflow
//@ needs-subprocess

use std::time::{Duration, Instant};

fn main() {
    let now = Instant::now();
    let _ = now + Duration::MAX;
}
