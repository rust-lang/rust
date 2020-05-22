// run-rustfix
#![warn(clippy::precedence)]
#![allow(unused_must_use, clippy::no_effect, clippy::unnecessary_operation)]
#![allow(clippy::identity_op)]
#![allow(clippy::eq_op)]

macro_rules! trip {
    ($a:expr) => {
        match $a & 0b1111_1111u8 {
            0 => println!("a is zero ({})", $a),
            _ => println!("a is {}", $a),
        }
    };
}

fn main() {
    1 << 2 + 3;
    1 + 2 << 3;
    4 >> 1 + 1;
    1 + 3 >> 2;
    1 ^ 1 - 1;
    3 | 2 - 1;
    3 & 5 - 2;
    -1i32.abs();
    -1f32.abs();

    // These should not trigger an error
    let _ = (-1i32).abs();
    let _ = (-1f32).abs();
    let _ = -(1i32).abs();
    let _ = -(1f32).abs();
    let _ = -(1i32.abs());
    let _ = -(1f32.abs());

    // Odd functions shoud not trigger an error
    let _ = -1f64.asin();
    let _ = -1f64.asinh();
    let _ = -1f64.atan();
    let _ = -1f64.atanh();
    let _ = -1f64.cbrt();
    let _ = -1f64.fract();
    let _ = -1f64.round();
    let _ = -1f64.signum();
    let _ = -1f64.sin();
    let _ = -1f64.sinh();
    let _ = -1f64.tan();
    let _ = -1f64.tanh();
    let _ = -1f64.to_degrees();
    let _ = -1f64.to_radians();

    let b = 3;
    trip!(b * 8);
}
