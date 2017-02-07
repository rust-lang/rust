#![feature(plugin)]
#![plugin(clippy)]

#![deny(invalid_upcast_comparisons)]
#![allow(unused, eq_op, no_effect, unnecessary_operation)]
fn main() {
    let zero: u32 = 0;
    let u8_max: u8 = 255;

    (u8_max as u32) > 300; //~ERROR because of the numeric bounds on `u8_max` prior to casting, this expression is always false
    (u8_max as u32) > 20;

    (zero as i32) < -5; //~ERROR because of the numeric bounds on `zero` prior to casting, this expression is always false
    (zero as i32) < 10;

    -5 < (zero as i32); //~ERROR because of the numeric bounds on `zero` prior to casting, this expression is always true
    0 <= (zero as i32); //~ERROR because of the numeric bounds on `zero` prior to casting, this expression is always true
    0 < (zero as i32);

    -5 > (zero as i32); //~ERROR because of the numeric bounds on `zero` prior to casting, this expression is always false
    -5 >= (u8_max as i32); //~ERROR because of the numeric bounds on `u8_max` prior to casting, this expression is always false
    1337 == (u8_max as i32); //~ERROR because of the numeric bounds on `u8_max` prior to casting, this expression is always false

    -5 == (zero as i32); //~ERROR because of the numeric bounds on `zero` prior to casting, this expression is always false
    -5 != (u8_max as i32); //~ERROR because of the numeric bounds on `u8_max` prior to casting, this expression is always true

    // Those are Ok:
    42 == (u8_max as i32);
    42 != (u8_max as i32);
    42 > (u8_max as i32);
    (u8_max as i32) == 42;
    (u8_max as i32) != 42;
    (u8_max as i32) > 42;
    (u8_max as i32) < 42;
}
