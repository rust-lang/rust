#![warn(clippy::manual_is_power_of_two)]

fn main() {
    let a = 16_u64;

    let _ = a.count_ones() == 1;
    let _ = a & (a - 1) == 0;

    // Test different orders of expression
    let _ = 1 == a.count_ones();
    let _ = (a - 1) & a == 0;
    let _ = 0 == a & (a - 1);
    let _ = 0 == (a - 1) & a;

    let b = 4_i64;

    // is_power_of_two only works for unsigned integers
    let _ = b.count_ones() == 1;
    let _ = b & (b - 1) == 0;
}
