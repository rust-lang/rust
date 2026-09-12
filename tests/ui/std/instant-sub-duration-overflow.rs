//! Regression test for <https://github.com/rust-lang/rust/issues/44216>.
//! Test overflowing `Instant` panics.
//@ run-fail
//@ error-pattern:overflow
//@ needs-subprocess

use std::time::{Instant, Duration};

fn main() {
    let now = Instant::now();
    let _ = now - Duration::from_secs(u64::MAX);
}
