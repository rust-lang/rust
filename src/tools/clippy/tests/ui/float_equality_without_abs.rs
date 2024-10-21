#![warn(clippy::float_equality_without_abs)]
//@no-rustfix: suggestions cause type ambiguity

// FIXME(f16_f128): add tests for these types when abs is available

pub fn is_roughly_equal(a: f32, b: f32) -> bool {
    (a - b) < f32::EPSILON
    //~^ ERROR: float equality check without `.abs()`
    //~| NOTE: `-D clippy::float-equality-without-abs` implied by `-D warnings`
}

pub fn main() {
    // all errors
    is_roughly_equal(1.0, 2.0);
    let a = 0.05;
    let b = 0.0500001;

    let _ = (a - b) < f32::EPSILON;
    //~^ ERROR: float equality check without `.abs()`
    let _ = a - b < f32::EPSILON;
    //~^ ERROR: float equality check without `.abs()`
    let _ = a - b.abs() < f32::EPSILON;
    //~^ ERROR: float equality check without `.abs()`
    let _ = (a as f64 - b as f64) < f64::EPSILON;
    //~^ ERROR: float equality check without `.abs()`
    let _ = 1.0 - 2.0 < f32::EPSILON;
    //~^ ERROR: float equality check without `.abs()`

    let _ = f32::EPSILON > (a - b);
    //~^ ERROR: float equality check without `.abs()`
    let _ = f32::EPSILON > a - b;
    //~^ ERROR: float equality check without `.abs()`
    let _ = f32::EPSILON > a - b.abs();
    //~^ ERROR: float equality check without `.abs()`
    let _ = f64::EPSILON > (a as f64 - b as f64);
    //~^ ERROR: float equality check without `.abs()`
    let _ = f32::EPSILON > 1.0 - 2.0;
    //~^ ERROR: float equality check without `.abs()`

    // those are correct
    let _ = (a - b).abs() < f32::EPSILON;
    let _ = (a as f64 - b as f64).abs() < f64::EPSILON;

    let _ = f32::EPSILON > (a - b).abs();
    let _ = f64::EPSILON > (a as f64 - b as f64).abs();
}
