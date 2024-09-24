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
