//@ run-fail
//@ error-pattern:overflow
//@ needs-subprocess

use std::time::{Duration, SystemTime};

fn main() {
    let now = SystemTime::now();
    let _ = now - Duration::from_secs(u64::MAX);
}
