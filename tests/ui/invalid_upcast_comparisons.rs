#![feature(plugin)]
#![plugin(clippy)]

#![deny(invalid_upcast_comparisons)]
#![allow(unused, eq_op, no_effect, unnecessary_operation)]
fn main() {
    let zero: u32 = 0;
    let u8_max: u8 = 255;

    (u8_max as u32) > 300;
    (u8_max as u32) > 20;

    (zero as i32) < -5;
    (zero as i32) < 10;

    -5 < (zero as i32);
    0 <= (zero as i32);
    0 < (zero as i32);

    -5 > (zero as i32);
    -5 >= (u8_max as i32);
    1337 == (u8_max as i32);

    -5 == (zero as i32);
    -5 != (u8_max as i32);

    // Those are Ok:
    42 == (u8_max as i32);
    42 != (u8_max as i32);
    42 > (u8_max as i32);
    (u8_max as i32) == 42;
    (u8_max as i32) != 42;
    (u8_max as i32) > 42;
    (u8_max as i32) < 42;
}
