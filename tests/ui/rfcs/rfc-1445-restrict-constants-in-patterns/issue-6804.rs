// Matching against NaN should result in a warning
// check-pass

#![allow(unused)]

const NAN: f64 = f64::NAN;

fn main() {
    let x = NAN;
    match x {
        NAN => {},
        //~^ WARN incorrect NaN comparison
        _ => {},
    };

    match [x, 1.0] {
        [NAN, _] => {},
        //~^ WARN incorrect NaN comparison
        _ => {},
    };
}
