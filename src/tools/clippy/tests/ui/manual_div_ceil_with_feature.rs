#![warn(clippy::manual_div_ceil)]
#![feature(int_roundings)]

fn main() {
    let x = 7_i32;
    let y = 4_i32;
    let z = 3_i32;
    let z_u: u32 = 11;

    // Lint.
    let _ = (x + (y - 1)) / y;
    let _ = ((y - 1) + x) / y;
    let _ = (x + y - 1) / y;

    let _ = (7_i32 + (4 - 1)) / 4;
    let _ = (7_i32 as u32 + (4 - 1)) / 4;
    let _ = (7_u32 as i32 + (4 - 1)) / 4;
    let _ = (z_u + (4 - 1)) / 4;

    // No lint.
    let _ = (x + (y - 2)) / y;
    let _ = (x + (y + 1)) / y;

    let _ = (x + (y - 1)) / z;
}

fn issue_13843() {
    let x = 3usize;
    let _ = (2048 + x - 1) / x;

    let x = 5usize;
    let _ = (2048usize + x - 1) / x;

    let x = 5usize;
    let _ = (2048_usize + x - 1) / x;

    let x = 2048usize;
    let _ = (x + 4 - 1) / 4;

    let _ = (2048 + 4 - 1) / 4;

    let _: u32 = (2048 + 6 - 1) / 6;
    let _: usize = (2048 + 6 - 1) / 6;
    let _: u32 = (0x2048 + 0x6 - 1) / 0x6;

    let _ = (2048 + 6u32 - 1) / 6u32;

    let x = -2;
    let _ = (-2048 + x - 1) / x;

    let _ = (1_000_000 + 6u32 - 1) / 6u32;
}
