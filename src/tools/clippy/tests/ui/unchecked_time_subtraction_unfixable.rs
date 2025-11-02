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
