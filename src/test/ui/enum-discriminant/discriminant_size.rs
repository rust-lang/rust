// run-pass
#![feature(core_intrinsics, repr128)]
//~^ WARN the feature `repr128` is incomplete

use std::intrinsics::discriminant_value;

enum E1 {
    A,
    B,
}

#[repr(i8)]
enum E2 {
    A = 7,
    B = -2,
}

#[repr(C)]
enum E3 {
    A = 42,
    B = 100,
}

#[repr(i128)]
enum E4 {
    A = 0x1223_3445_5667_7889,
    B = -0x1223_3445_5667_7889,
}

fn main() {
    let mut target: [isize; 3] = [0, 0, 0];
    target[1] = discriminant_value(&E1::A);
    assert_eq!(target, [0, 0, 0]);
    target[1] = discriminant_value(&E1::B);
    assert_eq!(target, [0, 1, 0]);

    let mut target: [i8; 3] = [0, 0, 0];
    target[1] = discriminant_value(&E2::A);
    assert_eq!(target, [0, 7, 0]);
    target[1] = discriminant_value(&E2::B);
    assert_eq!(target, [0, -2, 0]);

    let mut target: [isize; 3] = [0, 0, 0];
    target[1] = discriminant_value(&E3::A);
    assert_eq!(target, [0, 42, 0]);
    target[1] = discriminant_value(&E3::B);
    assert_eq!(target, [0, 100, 0]);

    let mut target: [i128; 3] = [0, 0, 0];
    target[1] = discriminant_value(&E4::A);
    assert_eq!(target, [0, 0x1223_3445_5667_7889, 0]);
    target[1] = discriminant_value(&E4::B);
    assert_eq!(target, [0, -0x1223_3445_5667_7889, 0]);
}
