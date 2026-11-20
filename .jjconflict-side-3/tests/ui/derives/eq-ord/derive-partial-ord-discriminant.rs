//! Regression test for https://github.com/rust-lang/rust/issues/15523

//@ run-pass
// Issue 15523: derive(PartialOrd) should use the provided
// discriminant values for the derived ordering.
//
// This is checking the basic functionality.

#[derive(PartialEq, PartialOrd)]
enum E1 {
    Pos2 = 2,
    Neg1 = -1,
    Pos1 = 1,
}

#[derive(PartialEq, PartialOrd)]
#[repr(u8)]
enum E2 {
    Pos2 = 2,
    PosMax = !0 as u8,
    Pos1 = 1,
}

#[derive(PartialEq, PartialOrd)]
#[repr(i8)]
enum E3 {
    Pos2 = 2,
    Neg1 = -1_i8,
    Pos1 = 1,
}

fn main() {
    assert!(E1::Pos2 > E1::Pos1);
    assert!(E1::Pos1 > E1::Neg1);
    assert!(E1::Pos2 > E1::Neg1);

    assert!(E2::Pos2 > E2::Pos1);
    assert!(E2::Pos1 < E2::PosMax);
    assert!(E2::Pos2 < E2::PosMax);

    assert!(E3::Pos2 > E3::Pos1);
    assert!(E3::Pos1 > E3::Neg1);
    assert!(E3::Pos2 > E3::Neg1);
}
