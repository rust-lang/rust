#![warn(clippy::integer_division)]

use std::num::NonZeroU32;

const TWO: NonZeroU32 = NonZeroU32::new(2).unwrap();

fn main() {
    let two = 2;
    let n = 1 / 2;
    //~^ integer_division

    let o = 1 / two;
    //~^ integer_division

    let p = two / 4;
    //~^ integer_division

    let x = 1. / 2.0;

    let a = 1;
    let s = a / TWO;
    //~^ integer_division
}
