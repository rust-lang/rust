// run-rustfix
#[warn(clippy::inconsistent_digit_grouping)]
#[allow(unused_variables, clippy::excessive_precision)]
fn main() {
    let good = (
        123,
        1_234,
        1_2345_6789,
        123_f32,
        1_234.12_f32,
        1_234.123_4_f32,
        1.123_456_7_f32,
    );
    let bad = (1_23_456, 1_234_5678, 1234_567, 1_234.5678_f32, 1.234_5678_f32);
}
