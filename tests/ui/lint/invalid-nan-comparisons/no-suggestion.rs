//@ check-pass

#![feature(f16, f128)]

fn main() {
    let mut effects = 0;
    let _ = 3.3 == {
        effects += 1;
        f64::NAN
    };
    //~^^^^ WARN incorrect NaN comparison
    let _ = effects;

    macro_rules! scoped_nan { () => {{ type Float = f64; Float::NAN }}; }
    let _ = 3.3 == scoped_nan!();
    //~^ WARN incorrect NaN comparison

    macro_rules! nan16 { () => { f16::NAN }; }
    let _ = 3.3 == nan16!();
    //~^ WARN incorrect NaN comparison
    macro_rules! nan128 { () => { f128::NAN }; }
    let _ = 3.3 != nan128!();
    //~^ WARN incorrect NaN comparison

    macro_rules! compare { () => { 3.3 != f64::NAN }; }
    //~^ WARN incorrect NaN comparison
    let _ = compare!();
}
