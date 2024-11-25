//@ run-fail
//@ check-run-results:overflow

use std::time::{Duration, Instant};

fn main() {
    let now = Instant::now();
    let _ = now + Duration::MAX;
}
