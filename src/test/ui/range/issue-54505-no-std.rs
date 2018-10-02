// error-pattern: `#[panic_handler]` function required, but not found
// error-pattern: language item required, but not found: `eh_personality`


// Regression test for #54505 - range borrowing suggestion had
// incorrect syntax (missing parentheses).

// This test doesn't use std
// (so all Ranges resolve to core::ops::Range...)

#![no_std]

use core::ops::RangeBounds;


// take a reference to any built-in range
fn take_range(_r: &impl RangeBounds<i8>) {}


fn main() {
    take_range(0..1);
    //~^ ERROR mismatched types [E0308]
    //~| HELP consider borrowing here
    //~| SUGGESTION &(0..1)

    take_range(1..);
    //~^ ERROR mismatched types [E0308]
    //~| HELP consider borrowing here
    //~| SUGGESTION &(1..)

    take_range(..);
    //~^ ERROR mismatched types [E0308]
    //~| HELP consider borrowing here
    //~| SUGGESTION &(..)

    take_range(0..=1);
    //~^ ERROR mismatched types [E0308]
    //~| HELP consider borrowing here
    //~| SUGGESTION &(0..=1)

    take_range(..5);
    //~^ ERROR mismatched types [E0308]
    //~| HELP consider borrowing here
    //~| SUGGESTION &(..5)

    take_range(..=42);
    //~^ ERROR mismatched types [E0308]
    //~| HELP consider borrowing here
    //~| SUGGESTION &(..=42)
}
