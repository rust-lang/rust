#![warn(clippy::float_cmp)]
#![allow(
    unused,
    clippy::no_effect,
    clippy::op_ref,
    clippy::unnecessary_operation,
    clippy::cast_lossless,
    clippy::many_single_char_names
)]

use std::ops::Add;

const ZERO: f32 = 0.0;
const ONE: f32 = ZERO + 1.0;

fn twice<T>(x: T) -> T
where
    T: Add<T, Output = T> + Copy,
{
    x + x
}

fn eq_fl(x: f32, y: f32) -> bool {
    if x.is_nan() {
        y.is_nan()
    } else {
        x == y
    } // no error, inside "eq" fn
}

fn fl_eq(x: f32, y: f32) -> bool {
    if x.is_nan() {
        y.is_nan()
    } else {
        x == y
    } // no error, inside "eq" fn
}

struct X {
    val: f32,
}

impl PartialEq for X {
    fn eq(&self, o: &X) -> bool {
        if self.val.is_nan() {
            o.val.is_nan()
        } else {
            self.val == o.val // no error, inside "eq" fn
        }
    }
}

fn main() {
    ZERO == 0f32; //no error, comparison with zero is ok
    1.0f32 != f32::INFINITY; // also comparison with infinity
    1.0f32 != f32::NEG_INFINITY; // and negative infinity
    ZERO == 0.0; //no error, comparison with zero is ok
    ZERO + ZERO != 1.0; //no error, comparison with zero is ok

    ONE == 1f32;
    ONE == 1.0 + 0.0;
    ONE + ONE == ZERO + ONE + ONE;
    ONE != 2.0;
    ONE != 0.0; // no error, comparison with zero is ok
    twice(ONE) != ONE;
    ONE as f64 != 2.0;
    ONE as f64 != 0.0; // no error, comparison with zero is ok

    let x: f64 = 1.0;

    x == 1.0;
    x != 0f64; // no error, comparison with zero is ok

    twice(x) != twice(ONE as f64);

    x < 0.0; // no errors, lower or greater comparisons need no fuzzyness
    x > 0.0;
    x <= 0.0;
    x >= 0.0;

    let xs: [f32; 1] = [0.0];
    let a: *const f32 = xs.as_ptr();
    let b: *const f32 = xs.as_ptr();

    assert_eq!(a, b); // no errors

    const ZERO_ARRAY: [f32; 2] = [0.0, 0.0];
    const NON_ZERO_ARRAY: [f32; 2] = [0.0, 0.1];

    let i = 0;
    let j = 1;

    ZERO_ARRAY[i] == NON_ZERO_ARRAY[j]; // ok, because lhs is zero regardless of i
    NON_ZERO_ARRAY[i] == NON_ZERO_ARRAY[j];

    let a1: [f32; 1] = [0.0];
    let a2: [f32; 1] = [1.1];

    a1 == a2;
    a1[0] == a2[0];

    // no errors - comparing signums is ok
    let x32 = 3.21f32;
    1.23f32.signum() == x32.signum();
    1.23f32.signum() == -(x32.signum());
    1.23f32.signum() == 3.21f32.signum();

    1.23f32.signum() != x32.signum();
    1.23f32.signum() != -(x32.signum());
    1.23f32.signum() != 3.21f32.signum();

    let x64 = 3.21f64;
    1.23f64.signum() == x64.signum();
    1.23f64.signum() == -(x64.signum());
    1.23f64.signum() == 3.21f64.signum();

    1.23f64.signum() != x64.signum();
    1.23f64.signum() != -(x64.signum());
    1.23f64.signum() != 3.21f64.signum();

    // the comparison should also look through references
    &0.0 == &ZERO;
    &&&&0.0 == &&&&ZERO;
}
