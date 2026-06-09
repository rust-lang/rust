//@ run-pass
#![feature(core_intrinsics)]

use std::intrinsics::discriminant_value;
use std::mem::size_of;

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

// Enums like this are found in the ecosystem, let's make sure they get the right size.
#[repr(C)]
#[allow(overflowing_literals)]
enum UnsignedIntEnum {
    A = 0,
    O = 0xffffffff, // doesn't fit into `int`, but fits into `unsigned int`
}

#[repr(i128)]
enum E4 {
    A = 0x1223_3445_5667_7889,
    B = -0x1223_3445_5667_7889,
}

fn main() {
    assert_eq!(size_of::<E1>(), 1);
    let mut target: [isize; 3] = [0, 0, 0];
    target[1] = discriminant_value(&E1::A);
    assert_eq!(target, [0, 0, 0]);
    target[1] = discriminant_value(&E1::B);
    assert_eq!(target, [0, 1, 0]);

    assert_eq!(size_of::<E2>(), 1);
    let mut target: [i8; 3] = [0, 0, 0];
    target[1] = discriminant_value(&E2::A);
    assert_eq!(target, [0, 7, 0]);
    target[1] = discriminant_value(&E2::B);
    assert_eq!(target, [0, -2, 0]);

    // E3's size is target-dependent
    let mut target: [isize; 3] = [0, 0, 0];
    target[1] = discriminant_value(&E3::A);
    assert_eq!(target, [0, 42, 0]);
    target[1] = discriminant_value(&E3::B);
    assert_eq!(target, [0, 100, 0]);

    #[allow(overflowing_literals)]
    {
        assert_eq!(size_of::<UnsignedIntEnum>(), 4);
        let mut target: [isize; 3] = [0, -1, 0];
        target[1] = discriminant_value(&UnsignedIntEnum::A);
        assert_eq!(target, [0, 0, 0]);
        target[1] = discriminant_value(&UnsignedIntEnum::O);
        assert_eq!(target, [0, 0xffffffff as isize, 0]);
    }

    assert_eq!(size_of::<E4>(), 16);
    let mut target: [i128; 3] = [0, 0, 0];
    target[1] = discriminant_value(&E4::A);
    assert_eq!(target, [0, 0x1223_3445_5667_7889, 0]);
    target[1] = discriminant_value(&E4::B);
    assert_eq!(target, [0, -0x1223_3445_5667_7889, 0]);
}
