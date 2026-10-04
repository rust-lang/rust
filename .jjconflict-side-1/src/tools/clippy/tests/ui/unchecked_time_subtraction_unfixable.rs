#![warn(clippy::unchecked_time_subtraction)]
//@no-rustfix

use std::time::{Duration, Instant};

fn main() {
    let dur1 = Duration::from_secs(5);
    let dur2 = Duration::from_secs(3);
    let dur3 = Duration::from_secs(1);

    // Chained Duration subtraction - should lint without suggestion due to complexity
    let _ = dur1 - dur2 - dur3;
    //~^ unchecked_time_subtraction
    //~| unchecked_time_subtraction

    // Chained Instant - Duration subtraction - should lint without suggestion due to complexity
    let instant1 = Instant::now();

    let _ = instant1 - dur2 - dur3;
    //~^ unchecked_time_subtraction
    //~| unchecked_time_subtraction
}

fn issue16499() {
    let _ = Duration::from_millis(1) - Duration::from_millis(2);
    //~^ unchecked_time_subtraction
    let _ = Duration::from_millis(1) - Duration::from_mins(2);
    //~^ unchecked_time_subtraction
    let _ = Duration::from_nanos(1) - Duration::from_secs(1);
    //~^ unchecked_time_subtraction
}
