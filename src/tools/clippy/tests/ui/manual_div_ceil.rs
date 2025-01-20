#![warn(clippy::manual_div_ceil)]

fn main() {
    let x = 7_u32;
    let y = 4_u32;
    let z = 11_u32;

    // Lint
    let _ = (x + (y - 1)) / y; //~ ERROR: manually reimplementing `div_ceil`
    let _ = ((y - 1) + x) / y; //~ ERROR: manually reimplementing `div_ceil`
    let _ = (x + y - 1) / y; //~ ERROR: manually reimplementing `div_ceil`

    let _ = (7_u32 + (4 - 1)) / 4; //~ ERROR: manually reimplementing `div_ceil`
    let _ = (7_i32 as u32 + (4 - 1)) / 4; //~ ERROR: manually reimplementing `div_ceil`

    // No lint
    let _ = (x + (y - 2)) / y;
    let _ = (x + (y + 1)) / y;

    let _ = (x + (y - 1)) / z;

    let x_i = 7_i32;
    let y_i = 4_i32;
    let z_i = 11_i32;

    // No lint because `int_roundings` feature is not enabled.
    let _ = (z as i32 + (y_i - 1)) / y_i;
    let _ = (7_u32 as i32 + (y_i - 1)) / y_i;
    let _ = (7_u32 as i32 + (4 - 1)) / 4;
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

    let _: u32 = (2048 + 6 - 1) / 6;
    let _: usize = (2048 + 6 - 1) / 6;
    let _: u32 = (0x2048 + 0x6 - 1) / 0x6;

    let _ = (2048 + 6u32 - 1) / 6u32;

    let _ = (1_000_000 + 6u32 - 1) / 6u32;
}
