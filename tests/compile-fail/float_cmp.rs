#![feature(plugin)]
#![plugin(clippy)]

#![deny(float_cmp)]
#![allow(unused)]

use std::ops::Add;

const ZERO : f32 = 0.0;
const ONE : f32 = ZERO + 1.0;

fn twice<T>(x : T) -> T where T : Add<T, Output = T>, T : Copy {
    x + x
}

fn eq_fl(x: f32, y: f32) -> bool {
    if x.is_nan() { y.is_nan() } else { x == y } // no error, inside "eq" fn
}

fn fl_eq(x: f32, y: f32) -> bool {
    if x.is_nan() { y.is_nan() } else { x == y } // no error, inside "eq" fn
}

struct X { val: f32 }

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
    1.0f32 != ::std::f32::INFINITY; // also comparison with infinity
    1.0f32 != ::std::f32::NEG_INFINITY; // and negative infinity
    ZERO == 0.0; //no error, comparison with zero is ok
    ZERO + ZERO != 1.0; //no error, comparison with zero is ok

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
