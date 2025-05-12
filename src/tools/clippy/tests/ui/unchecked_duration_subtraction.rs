#![warn(clippy::unchecked_duration_subtraction)]

use std::time::{Duration, Instant};

fn main() {
    let _first = Instant::now();
    let second = Duration::from_secs(3);

    let _ = _first - second;
    //~^ unchecked_duration_subtraction

    let _ = Instant::now() - Duration::from_secs(5);
    //~^ unchecked_duration_subtraction

    let _ = _first - Duration::from_secs(5);
    //~^ unchecked_duration_subtraction

    let _ = Instant::now() - second;
    //~^ unchecked_duration_subtraction
}
