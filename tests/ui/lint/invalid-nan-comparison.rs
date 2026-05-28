//@ check-pass

#![feature(f16, f128)]

fn main() {
    f16();
    f32();
    f64();
    f128();
}

const TEST: bool = 5f32 == f32::NAN;
//~^ WARN incorrect NaN comparison

fn f16() {
    macro_rules! number { () => { 5f16 }; }
    let x = number!();
    x == f16::NAN;
    //~^ WARN incorrect NaN comparison
    x != f16::NAN;
    //~^ WARN incorrect NaN comparison
    x < f16::NAN;
    //~^ WARN incorrect NaN comparison
    x > f16::NAN;
    //~^ WARN incorrect NaN comparison
    x <= f16::NAN;
    //~^ WARN incorrect NaN comparison
    x >= f16::NAN;
    //~^ WARN incorrect NaN comparison
    number!() == f16::NAN;
    //~^ WARN incorrect NaN comparison
    f16::NAN != number!();
    //~^ WARN incorrect NaN comparison
}

fn f32() {
    macro_rules! number { () => { 5f32 }; }
    let x = number!();
    x == f32::NAN;
    //~^ WARN incorrect NaN comparison
    x != f32::NAN;
    //~^ WARN incorrect NaN comparison
    x < f32::NAN;
    //~^ WARN incorrect NaN comparison
    x > f32::NAN;
    //~^ WARN incorrect NaN comparison
    x <= f32::NAN;
    //~^ WARN incorrect NaN comparison
    x >= f32::NAN;
    //~^ WARN incorrect NaN comparison
    number!() == f32::NAN;
    //~^ WARN incorrect NaN comparison
    f32::NAN != number!();
    //~^ WARN incorrect NaN comparison
}

fn f64() {
    macro_rules! number { () => { 5f64 }; }
    let x = number!();
    x == f64::NAN;
    //~^ WARN incorrect NaN comparison
    x != f64::NAN;
    //~^ WARN incorrect NaN comparison
    x < f64::NAN;
    //~^ WARN incorrect NaN comparison
    x > f64::NAN;
    //~^ WARN incorrect NaN comparison
    x <= f64::NAN;
    //~^ WARN incorrect NaN comparison
    x >= f64::NAN;
    //~^ WARN incorrect NaN comparison
    number!() == f64::NAN;
    //~^ WARN incorrect NaN comparison
    f64::NAN != number!();
    //~^ WARN incorrect NaN comparison
}

fn f128() {
    macro_rules! number { () => { 5f128 }; }
    let x = number!();
    x == f128::NAN;
    //~^ WARN incorrect NaN comparison
    x != f128::NAN;
    //~^ WARN incorrect NaN comparison
    x < f128::NAN;
    //~^ WARN incorrect NaN comparison
    x > f128::NAN;
    //~^ WARN incorrect NaN comparison
    x <= f128::NAN;
    //~^ WARN incorrect NaN comparison
    x >= f128::NAN;
    //~^ WARN incorrect NaN comparison
    number!() == f128::NAN;
    //~^ WARN incorrect NaN comparison
    f128::NAN != number!();
    //~^ WARN incorrect NaN comparison
}
