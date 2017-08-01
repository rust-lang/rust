#![feature(plugin)]
#![plugin(clippy)]
#[warn(unreadable_literal)]
#[allow(unused_variables)]
fn main() {
    let good = (0b1011_i64, 0o1_234_u32, 0x1_234_567, 1_2345_6789, 1234_f32, 1_234.12_f32, 1_234.123_f32, 1.123_4_f32);
    let bad = (0b10110_i64, 0x12345678901_usize, 12345_f32, 1.23456_f32);
}
