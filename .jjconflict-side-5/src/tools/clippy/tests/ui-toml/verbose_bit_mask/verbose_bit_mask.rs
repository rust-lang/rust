#![warn(clippy::verbose_bit_mask)]
fn main() {
    let v: i32 = 0;
    let _ = v & 0b11111 == 0;
    let _ = v & 0b111111 == 0;
    //~^ ERROR: bit mask could be simplified
}
