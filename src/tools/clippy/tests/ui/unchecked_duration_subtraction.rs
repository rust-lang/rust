// run-rustfix
#![warn(clippy::unchecked_duration_subtraction)]

use std::time::{Duration, Instant};

fn main() {
    let _first = Instant::now();
    let second = Duration::from_secs(3);

    let _ = _first - second;

    let _ = Instant::now() - Duration::from_secs(5);

    let _ = _first - Duration::from_secs(5);

    let _ = Instant::now() - second;
}
