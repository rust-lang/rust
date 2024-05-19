//@ run-fail
//@ error-pattern:overflow

use std::time::{Duration, Instant};

fn main() {
    let now = Instant::now();
    let _ = now + Duration::MAX;
}
