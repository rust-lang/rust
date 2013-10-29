// Matching against NaN should result in a warning

use std::f64::NAN;

fn main() {
    let x = NAN;
    match x {
        NAN => {},
        _ => {},
    };
    //~^^^ WARNING unmatchable NaN in pattern, use the is_nan method in a guard instead
    match [x, 1.0] {
        [NAN, _] => {},
        _ => {},
    };
    //~^^^ WARNING unmatchable NaN in pattern, use the is_nan method in a guard instead
}

// At least one error is needed so that compilation fails
#[static_assert]
static b: bool = false; //~ ERROR static assertion failed
