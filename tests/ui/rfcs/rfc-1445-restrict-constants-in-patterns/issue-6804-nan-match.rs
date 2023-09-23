// Matching against NaN should result in an error
#![feature(exclusive_range_pattern)]
#![allow(unused)]

const NAN: f64 = f64::NAN;

#[derive(PartialEq, Eq)]
struct MyType<T>(T);

const C: MyType<f32> = MyType(f32::NAN);

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

    match MyType(1.0f32) {
        C => {}, //~ ERROR cannot use NaN in patterns
        _ => {},
    }

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
