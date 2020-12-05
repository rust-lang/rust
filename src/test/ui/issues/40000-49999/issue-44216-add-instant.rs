// run-fail
// error-pattern:overflow
// ignore-emscripten no processes

use std::time::{Instant, Duration};

fn main() {
    let now = Instant::now();
    let _ = now + Duration::from_secs(u64::MAX);
}
