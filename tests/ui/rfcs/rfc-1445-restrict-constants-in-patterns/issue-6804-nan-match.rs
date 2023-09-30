// Matching against NaN should result in an error
#![feature(exclusive_range_pattern)]
#![allow(unused)]
#![allow(illegal_floating_point_literal_pattern)]

const NAN: f64 = f64::NAN;

fn main() {
    let x = NAN;
    match x {
        NAN => {}, //~ ERROR cannot use NaN in patterns
        _ => {},
    };

    match [x, 1.0] {
        [NAN, _] => {}, //~ ERROR cannot use NaN in patterns
        _ => {},
    };

    // Also cover range patterns
    match x {
        NAN..=1.0 => {}, //~ ERROR cannot use NaN in patterns
        //~^ ERROR lower range bound must be less than or equal to upper
        -1.0..=NAN => {}, //~ ERROR cannot use NaN in patterns
        //~^ ERROR lower range bound must be less than or equal to upper
        NAN.. => {}, //~ ERROR cannot use NaN in patterns
        //~^ ERROR lower range bound must be less than or equal to upper
        ..NAN => {}, //~ ERROR cannot use NaN in patterns
        //~^ ERROR lower range bound must be less than upper
        _ => {},
    };
}
