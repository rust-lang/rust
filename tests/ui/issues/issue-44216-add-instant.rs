//@ run-fail
//@ check-run-results

use std::time::{Duration, Instant};

fn main() {
    let now = Instant::now();
    let _ = now + Duration::MAX;
}
