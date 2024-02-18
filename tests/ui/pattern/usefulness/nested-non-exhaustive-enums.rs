//@ aux-build:non-exhaustive.rs

extern crate non_exhaustive;

use non_exhaustive::NonExhaustiveEnum;

fn main() {
    match Some(NonExhaustiveEnum::A) {
        //~^ ERROR non-exhaustive patterns: `Some(_)` not covered [E0004]
        //~| NOTE pattern `Some(_)` not covered
        //~| NOTE `Option<NonExhaustiveEnum>` defined here
        //~| NOTE the matched value is of type `Option<NonExhaustiveEnum>`
        //~| NOTE `NonExhaustiveEnum` is marked as non-exhaustive
        Some(NonExhaustiveEnum::A) => {}
        Some(NonExhaustiveEnum::B) => {}
        None => {}
    }
}
