#![feature(plugin)]
#![plugin(clippy)]

#![deny(invalid_upcast_comparisons)]
#![allow(unused, eq_op, no_effect)]
fn main() {
    let zero: u32 = 0;
    let u8_max: u8 = 255;

    (u8_max as u32) > 300; //~ERROR
    (u8_max as u32) > 20;

    (zero as i32) < -5; //~ERROR
    (zero as i32) < 10;
}
