#![warn(clippy::default_numeric_fallback)]
#![allow(unused)]

fn main() {
    // Bad.
    let x = 1;
    let x = 0.1;
    let x = if true { 1 } else { 2 };

    // Good.
    let x = 1_i32;
    let x: i32 = 1;
    let x: _ = 1;
    let x = 0.1_f64;
    let x: f64 = 0.1;
    let x: _ = 0.1;
    let x: _ = if true { 1 } else { 2 };
}
