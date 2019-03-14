#![warn(clippy::float_cmp_const)]
#![allow(clippy::float_cmp)]
#![allow(unused, clippy::no_effect, clippy::unnecessary_operation)]

const ONE: f32 = 1.0;
const TWO: f32 = 2.0;

fn eq_one(x: f32) -> bool {
    if x.is_nan() {
        false
    } else {
        x == ONE
    } // no error, inside "eq" fn
}

fn main() {
    // has errors
    1f32 == ONE;
    TWO == ONE;
    TWO != ONE;
    ONE + ONE == TWO;
    let x = 1;
    x as f32 == ONE;

    let v = 0.9;
    v == ONE;
    v != ONE;

    // no errors, lower than or greater than comparisons
    v < ONE;
    v > ONE;
    v <= ONE;
    v >= ONE;

    // no errors, zero and infinity values
    ONE != 0f32;
    TWO == 0f32;
    ONE != ::std::f32::INFINITY;
    ONE == ::std::f32::NEG_INFINITY;

    // no errors, but will warn clippy::float_cmp if '#![allow(float_cmp)]' above is removed
    let w = 1.1;
    v == w;
    v != w;
    v == 1.0;
    v != 1.0;
}
