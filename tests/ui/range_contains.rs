// run-rustfix

#[warn(clippy::manual_range_contains)]
#[allow(unused)]
#[allow(clippy::no_effect)]
#[allow(clippy::short_circuit_statement)]
#[allow(clippy::unnecessary_operation)]
fn main() {
    let x = 9_u32;

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
}

// Fix #6373
pub const fn in_range(a: i32) -> bool {
    3 <= a && a <= 20
}
