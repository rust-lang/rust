//@ run-fail
//@ error-pattern:overflow
//@ ignore-emscripten no processes

use std::time::{Duration, SystemTime};

fn main() {
    let now = SystemTime::now();
    let _ = now - Duration::from_secs(u64::MAX);
}
