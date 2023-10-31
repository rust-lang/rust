// check-pass

fn main() {
    f32();
    f64();
}

const TEST: bool = 5f32 == f32::NAN;
//~^ WARN incorrect NaN comparison

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
