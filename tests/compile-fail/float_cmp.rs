#![feature(plugin)]
#![plugin(clippy)]

#![deny(float_cmp)]
#![allow(unused, no_effect, unnecessary_operation)]

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

    ONE == 1f32;
    //~^ ERROR strict comparison of f32 or f64
    //~| HELP within some error
    //~| SUGGESTION (ONE - 1f32).abs() < error
    ONE == (1.0 + 0.0);
    //~^ ERROR strict comparison of f32 or f64
    //~| HELP within some error
    //~| SUGGESTION (ONE - (1.0 + 0.0)).abs() < error

    ONE + ONE == (ZERO + ONE + ONE);
    //~^ ERROR strict comparison of f32 or f64
    //~| HELP within some error
    //~| SUGGESTION (ONE + ONE - (ZERO + ONE + ONE)).abs() < error

    ONE != 2.0;
    //~^ ERROR strict comparison of f32 or f64
    //~| HELP within some error
    //~| SUGGESTION (ONE - 2.0).abs() < error
    ONE != 0.0; // no error, comparison with zero is ok
    twice(ONE) != ONE;
    //~^ ERROR strict comparison of f32 or f64
    //~| HELP within some error
    //~| SUGGESTION (twice(ONE) - ONE).abs() < error
    ONE as f64 != 2.0;
    //~^ ERROR strict comparison of f32 or f64
    //~| HELP within some error
    //~| SUGGESTION (ONE as f64 - 2.0).abs() < error
    ONE as f64 != 0.0; // no error, comparison with zero is ok

    let x : f64 = 1.0;

    x == 1.0;
    //~^ ERROR strict comparison of f32 or f64
    //~| HELP within some error
    //~| SUGGESTION (x - 1.0).abs() < error
    x != 0f64; // no error, comparison with zero is ok

    twice(x) != twice(ONE as f64);
    //~^ ERROR strict comparison of f32 or f64
    //~| HELP within some error
    //~| SUGGESTION (twice(x) - twice(ONE as f64)).abs() < error


    x < 0.0; // no errors, lower or greater comparisons need no fuzzyness
    x > 0.0;
    x <= 0.0;
    x >= 0.0;

    let xs : [f32; 1] = [0.0];
    let a: *const f32 = xs.as_ptr();
    let b: *const f32 = xs.as_ptr();

    assert!(a == b); // no errors
}
