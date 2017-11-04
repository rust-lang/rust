


#![warn(float_cmp_const)]
#![allow(unused, no_effect, unnecessary_operation)]

const ONE: f32 = 1.0;
const TWO: f32 = 2.0;

fn eq_one(x: f32) -> bool {
    if x.is_nan() { false } else { x == ONE } // no error, inside "eq" fn
}

fn main() {
    // has errors
    1f32 == ONE;
    TWO == ONE;
    TWO != ONE;
    ONE + ONE == TWO;
    1 as f32 == ONE;

    let v = 0.9;
    v == ONE;
    v != ONE;

    // no errors, lower than or greater than comparisons
    v < ONE;
    v > ONE;
    v <= ONE;
    v >= ONE;
}
