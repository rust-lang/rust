// check-pass
// run-rustfix

fn main() {
    let x = 5f32;
    let _ = x == f32::NAN;
    //~^ WARN incorrect NaN comparison
    let _ = x != f32::NAN;
    //~^ WARN incorrect NaN comparison

    let x = 5f64;
    let _ = x == f64::NAN;
    //~^ WARN incorrect NaN comparison
    let _ = x != f64::NAN;
    //~^ WARN incorrect NaN comparison

    let b = &2.3f32;
    if b != &f32::NAN {}
    //~^ WARN incorrect NaN comparison

    let b = &2.3f32;
    if b != { &f32::NAN } {}
    //~^ WARN incorrect NaN comparison

    let _ =
        b != {
    //~^ WARN incorrect NaN comparison
            &f32::NAN
        };

    #[allow(unused_macros)]
    macro_rules! nan { () => { f32::NAN }; }
    macro_rules! number { () => { 5f32 }; }

    let _ = nan!() == number!();
    //~^ WARN incorrect NaN comparison
    let _ = number!() != nan!();
    //~^ WARN incorrect NaN comparison
}
