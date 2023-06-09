#![deny(arithmetic_overflow)]
fn main() {
    let _x = -1_i32 >> -1;
    let _y = 1u32 >> 10000000000000u32;
}
