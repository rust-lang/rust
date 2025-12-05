//@no-rustfix: suggestions have an error margin placeholder
#![warn(clippy::float_cmp_const)]
#![allow(clippy::float_cmp)]
#![allow(unused, clippy::no_effect, clippy::unnecessary_operation)]

const ONE: f32 = 1.0;
const TWO: f32 = 2.0;

fn eq_one(x: f32) -> bool {
    if x.is_nan() { false } else { x == ONE } // no error, inside "eq" fn
}

fn main() {
    // has errors
    1f32 == ONE;
    //~^ float_cmp_const

    TWO == ONE;
    //~^ float_cmp_const

    TWO != ONE;
    //~^ float_cmp_const

    ONE + ONE == TWO;
    //~^ float_cmp_const

    let x = 1;
    x as f32 == ONE;
    //~^ float_cmp_const

    let v = 0.9;
    v == ONE;
    //~^ float_cmp_const

    v != ONE;
    //~^ float_cmp_const

    // no errors, lower than or greater than comparisons
    v < ONE;
    v > ONE;
    v <= ONE;
    v >= ONE;

    // no errors, zero and infinity values
    ONE != 0f32;
    TWO == 0f32;
    ONE != f32::INFINITY;
    ONE == f32::NEG_INFINITY;

    // no errors, but will warn clippy::float_cmp if '#![allow(float_cmp)]' above is removed
    let w = 1.1;
    v == w;
    v != w;
    v == 1.0;
    v != 1.0;

    const ZERO_ARRAY: [f32; 3] = [0.0, 0.0, 0.0];
    const ZERO_INF_ARRAY: [f32; 3] = [0.0, f32::INFINITY, f32::NEG_INFINITY];
    const NON_ZERO_ARRAY: [f32; 3] = [0.0, 0.1, 0.2];
    const NON_ZERO_ARRAY2: [f32; 3] = [0.2, 0.1, 0.0];

    // no errors, zero and infinity values
    NON_ZERO_ARRAY[0] == NON_ZERO_ARRAY2[1]; // lhs is 0.0
    ZERO_ARRAY == NON_ZERO_ARRAY; // lhs is all zeros
    ZERO_INF_ARRAY == NON_ZERO_ARRAY; // lhs is all zeros or infinities

    // has errors
    NON_ZERO_ARRAY == NON_ZERO_ARRAY2;
    //~^ float_cmp_const
}
