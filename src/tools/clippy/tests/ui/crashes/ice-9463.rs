#![deny(arithmetic_overflow)]
fn main() {
    let _x = -1_i32 >> -1;
    //~^ ERROR: this arithmetic operation will overflow
    let _y = 1u32 >> 10000000000000u32;
    //~^ ERROR: this arithmetic operation will overflow
    //~| ERROR: literal out of range
}
