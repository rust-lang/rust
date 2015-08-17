#![feature(plugin)]
#![plugin(clippy)]

use std::ops::Add;

const ZERO : f32 = 0.0;
const ONE : f32 = ZERO + 1.0;

fn twice<T>(x : T) -> T where T : Add<T, Output = T>, T : Copy {
    x + x
}

#[deny(float_cmp)]
#[allow(unused)]
fn main() {
    ZERO == 0f32; //no error, comparison with zero is ok
    ZERO == 0.0; //no error, comparison with zero is ok
    ZERO + ZERO != 1.0; //~ERROR !=-comparison of f32 or f64

    ONE == 1f32; //~ERROR ==-comparison of f32 or f64
    ONE == (1.0 + 0.0); //~ERROR ==-comparison of f32 or f64

    ONE + ONE == (ZERO + ONE + ONE); //~ERROR ==-comparison of f32 or f64

    ONE != 2.0; //~ERROR !=-comparison of f32 or f64
    ONE != 0.0; // no error, comparison with zero is ok
    twice(ONE) != ONE; //~ERROR !=-comparison of f32 or f64
    ONE as f64 != 2.0; //~ERROR !=-comparison of f32 or f64
    ONE as f64 != 0.0; // no error, comparison with zero is ok

    let x : f64 = 1.0;

    x == 1.0; //~ERROR ==-comparison of f32 or f64
    x != 0f64; // no error, comparison with zero is ok

    twice(x) != twice(ONE as f64); //~ERROR !=-comparison of f32 or f64


    x < 0.0; // no errors, lower or greater comparisons need no fuzzyness
    x > 0.0;
    x <= 0.0;
    x >= 0.0;
}
