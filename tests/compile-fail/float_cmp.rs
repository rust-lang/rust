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
    ZERO == 0f32; //~ERROR
    ZERO == 0.0; //~ERROR
    ZERO + ZERO != 1.0; //~ERROR

    ONE != 0.0; //~ERROR
    twice(ONE) != ONE; //~ERROR
    ONE as f64 != 0.0; //~ERROR

    let x : f64 = 1.0;

    x == 1.0; //~ERROR
    x != 0f64; //~ERROR

    twice(x) != twice(ONE as f64); //~ERROR

    x < 0.0;
    x > 0.0;
    x <= 0.0;
    x >= 0.0;
}
