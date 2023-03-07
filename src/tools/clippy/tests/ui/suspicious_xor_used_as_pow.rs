#![allow(unused)]
#![warn(clippy::suspicious_xor_used_as_pow)]
#![allow(clippy::eq_op)]

macro_rules! macro_test {
    () => {
        13
    };
}

macro_rules! macro_test_inside {
    () => {
        1 ^ 2 // should warn even if inside macro
    };
}

fn main() {
    // Should warn:
    let _ = 2 ^ 5;
    let _ = 2i32 ^ 9i32;
    let _ = 2i32 ^ 2i32;
    let _ = 50i32 ^ 3i32;
    let _ = 5i32 ^ 8i32;
    let _ = 2i32 ^ 32i32;
    macro_test_inside!();

    // Should not warn:
    let x = 0x02;
    let _ = x ^ 2;
    let _ = 2 ^ x;
    let _ = x ^ 5;
    let _ = 10 ^ 0b0101;
    let _ = 2i32 ^ macro_test!();
}
