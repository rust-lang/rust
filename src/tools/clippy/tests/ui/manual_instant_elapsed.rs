// run-rustfix
#![warn(clippy::manual_instant_elapsed)]
#![allow(clippy::unnecessary_operation)]
#![allow(clippy::unchecked_duration_subtraction)]
#![allow(unused_variables)]
#![allow(unused_must_use)]

use std::time::Instant;

fn main() {
    let prev_instant = Instant::now();

    {
        // don't influence
        let another_instant = Instant::now();
    }

    let duration = Instant::now() - prev_instant;

    // don't catch
    let duration = prev_instant.elapsed();

    Instant::now() - duration;

    let ref_to_instant = &Instant::now();

    Instant::now() - *ref_to_instant; // to ensure parens are added correctly
}
