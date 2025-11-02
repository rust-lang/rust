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

    // Duration - Duration cases
    let dur1 = Duration::from_secs(5);
    let dur2 = Duration::from_secs(3);

    let _ = dur1 - dur2;
    //~^ unchecked_time_subtraction

    let _ = Duration::from_secs(10) - Duration::from_secs(5);
    //~^ unchecked_time_subtraction

    let _ = second - dur1;
    //~^ unchecked_time_subtraction

    // Duration multiplication and subtraction
    let _ = 2 * dur1 - dur2;
    //~^ unchecked_time_subtraction
}
