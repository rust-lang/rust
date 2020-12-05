// Matching against NaN should result in a warning

#![allow(unused)]
#![deny(illegal_floating_point_literal_pattern)]

const NAN: f64 = f64::NAN;

fn main() {
    let x = NAN;
    match x {
        NAN => {}, //~ ERROR floating-point types cannot be used
        //~^ WARN this was previously accepted by the compiler but is being phased out
        //~| ERROR floating-point types cannot be used in patterns
        //~| WARN this was previously accepted by the compiler but is being phased out
        _ => {},
    };

    match [x, 1.0] {
        [NAN, _] => {}, //~ ERROR floating-point types cannot be used
                        //~| ERROR floating-point types cannot be used
        //~| WARN this was previously accepted by the compiler but is being phased out
        //~| WARN this was previously accepted by the compiler but is being phased out
        _ => {},
    };
}
