// run-rustfix

#[warn(clippy::unreadable_literal)]
#[allow(unused_variables)]
fn main() {
    let good = (
        0b1011_i64,
        0o1_234_u32,
        0x1_234_567,
        65536,
        1_2345_6789,
        1234_f32,
        1_234.12_f32,
        1_234.123_f32,
        1.123_4_f32,
    );
    let bad = (0b110110_i64, 0x12345678901_usize, 123456_f32, 1.234567_f32);
    let good_sci = 1.1234e1;
    let bad_sci = 1.123456e1;

    let fail9 = 0xabcdef;
    let fail10: u32 = 0xBAFEBAFE;
    let fail11 = 0xabcdeff;
    let fail12: i128 = 0xabcabcabcabcabcabc;
}
