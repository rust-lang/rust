// run-rustfix
#[warn(clippy::inconsistent_digit_grouping)]
#[deny(clippy::unreadable_literal)]
#[allow(unused_variables, clippy::excessive_precision)]
fn main() {
    macro_rules! mac1 {
        () => {
            1_23_456
        };
    }
    macro_rules! mac2 {
        () => {
            1_234.5678_f32
        };
    }

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

    // Test padding
    let _ = 0x100000;
    let _ = 0x1000000;
    let _ = 0x10000000;
    let _ = 0x100000000_u64;

    // Test suggestion when fraction has no digits
    let _: f32 = 1_23_456.;

    // Test UUID formatted literal
    let _: u128 = 0x12345678_1234_1234_1234_123456789012;

    // Ignore literals in macros
    let _ = mac1!();
    let _ = mac2!();

    // Issue #6096
    // Allow separating exponent with '_'
    let _ = 1.025_011_10_E0;
}
