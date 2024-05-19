#![warn(clippy::is_digit_ascii_radix)]

const TEN: u32 = 10;

fn main() {
    let c: char = '6';

    // Should trigger the lint.
    let _ = c.is_digit(10);
    let _ = c.is_digit(16);
    let _ = c.is_digit(0x10);

    // Should not trigger the lint.
    let _ = c.is_digit(11);
    let _ = c.is_digit(TEN);
}
