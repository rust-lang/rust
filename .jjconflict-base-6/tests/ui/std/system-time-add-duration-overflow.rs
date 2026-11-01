//! Regression test for <https://github.com/rust-lang/rust/issues/44216>.
//! Test overflowing `SystemTime` panics.
//@ run-fail
//@ error-pattern:overflow
//@ needs-subprocess

use std::time::{Duration, SystemTime};

fn main() {
    let now = SystemTime::now();
    let _ = now + Duration::from_secs(u64::MAX);
}
