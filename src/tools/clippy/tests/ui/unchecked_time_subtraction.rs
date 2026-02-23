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

fn issue16230() {
    use std::ops::Sub as _;

    Duration::ZERO.sub(Duration::MAX);
    //~^ unchecked_time_subtraction

    let _ = Duration::ZERO - Duration::MAX;
    //~^ unchecked_time_subtraction
}

fn issue16234() {
    use std::ops::Sub as _;

    macro_rules! duration {
        ($secs:expr) => {
            Duration::from_secs($secs)
        };
    }

    duration!(0).sub(duration!(1));
    //~^ unchecked_time_subtraction
    let _ = duration!(0) - duration!(1);
    //~^ unchecked_time_subtraction
}
