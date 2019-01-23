// run-rustfix
#[warn(clippy::large_digit_groups)]
#[allow(unused_variables)]
fn main() {
    let good = (
        0b1011_i64,
        0o1_234_u32,
        0x1_234_567,
        1_2345_6789,
        1234_f32,
        1_234.12_f32,
        1_234.123_f32,
        1.123_4_f32,
    );
    let bad = (
        0b1_10110_i64,
        0x1_23456_78901_usize,
        1_23456_f32,
        1_23456.12_f32,
        1_23456.12345_f64,
        1_23456.12345_6_f64,
    );
}
