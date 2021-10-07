#![warn(clippy::modulo_arithmetic)]
#![allow(clippy::no_effect, clippy::unnecessary_operation, clippy::modulo_one)]

fn main() {
    // Lint when both sides are const and of the opposite sign
    -1 % 2;
    1 % -2;
    (1 - 2) % (1 + 2);
    (1 + 2) % (1 - 2);
    35 * (7 - 4 * 2) % (-500 * -600);

    -1i8 % 2i8;
    1i8 % -2i8;
    -1i16 % 2i16;
    1i16 % -2i16;
    -1i32 % 2i32;
    1i32 % -2i32;
    -1i64 % 2i64;
    1i64 % -2i64;
    -1i128 % 2i128;
    1i128 % -2i128;
    -1isize % 2isize;
    1isize % -2isize;

    // No lint when both sides are const and of the same sign
    1 % 2;
    -1 % -2;
    (1 + 2) % (-1 + 2);
    (-1 - 2) % (1 - 2);

    1u8 % 2u8;
    1u16 % 2u16;
    1u32 % 2u32;
    1u64 % 2u64;
    1u128 % 2u128;
    1usize % 2usize;
}
