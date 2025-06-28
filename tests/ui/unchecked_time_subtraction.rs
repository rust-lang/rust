#![warn(clippy::unchecked_time_subtraction)]

use std::time::{Duration, Instant};

fn main() {
    let _first = Instant::now();
    let second = Duration::from_secs(3);

    let _ = _first - second;
    //~^ unchecked_time_subtraction

    let _ = Instant::now() - Duration::from_secs(5);
    //~^ unchecked_time_subtraction

    let _ = _first - Duration::from_secs(5);
    //~^ unchecked_time_subtraction

    let _ = Instant::now() - second;
    //~^ unchecked_time_subtraction
}
