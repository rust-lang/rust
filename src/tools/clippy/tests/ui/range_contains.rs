// run-rustfix

#![warn(clippy::manual_range_contains)]
#![allow(unused)]
#![allow(clippy::no_effect)]
#![allow(clippy::short_circuit_statement)]
#![allow(clippy::unnecessary_operation)]

fn main() {
    let x = 9_i32;

    // order shouldn't matter
    x >= 8 && x < 12;
    x < 42 && x >= 21;
    100 > x && 1 <= x;

    // also with inclusive ranges
    x >= 9 && x <= 99;
    x <= 33 && x >= 1;
    999 >= x && 1 <= x;

    // and the outside
    x < 8 || x >= 12;
    x >= 42 || x < 21;
    100 <= x || 1 > x;

    // also with the outside of inclusive ranges
    x < 9 || x > 99;
    x > 33 || x < 1;
    999 < x || 1 > x;

    // not a range.contains
    x > 8 && x < 12; // lower bound not inclusive
    x < 8 && x <= 12; // same direction
    x >= 12 && 12 >= x; // same bounds
    x < 8 && x > 12; // wrong direction

    x <= 8 || x >= 12;
    x >= 8 || x >= 12;
    x < 12 || 12 < x;
    x >= 8 || x <= 12;

    // Fix #6315
    let y = 3.;
    y >= 0. && y < 1.;
    y < 0. || y > 1.;

    // handle negatives #8721
    x >= -10 && x <= 10;
    x >= 10 && x <= -10;
    y >= -3. && y <= 3.;
    y >= 3. && y <= -3.;

    // Fix #8745
    let z = 42;
    (x >= 0) && (x <= 10) && (z >= 0) && (z <= 10);
    (x < 0) || (x >= 10) || (z < 0) || (z >= 10);
    // Make sure operators in parens don't give a breaking suggestion
    ((x % 2 == 0) || (x < 0)) || (x >= 10);
}

// Fix #6373
pub const fn in_range(a: i32) -> bool {
    3 <= a && a <= 20
}

#[clippy::msrv = "1.34"]
fn msrv_1_34() {
    let x = 5;
    x >= 8 && x < 34;
}

#[clippy::msrv = "1.35"]
fn msrv_1_35() {
    let x = 5;
    x >= 8 && x < 35;
}
