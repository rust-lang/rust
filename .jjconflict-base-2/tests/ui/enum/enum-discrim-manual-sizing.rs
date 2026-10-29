//@ run-pass
#![allow(dead_code)]

use std::mem::{size_of, align_of};

#[repr(i8)]
enum Ei8 {
    Ai8 = 0,
    Bi8 = 1
}

#[repr(u8)]
enum Eu8 {
    Au8 = 0,
    Bu8 = 1
}

#[repr(i16)]
enum Ei16 {
    Ai16 = 0,
    Bi16 = 1
}

#[repr(u16)]
enum Eu16 {
    Au16 = 0,
    Bu16 = 1
}

#[repr(i32)]
enum Ei32 {
    Ai32 = 0,
    Bi32 = 1
}

#[repr(u32)]
enum Eu32 {
    Au32 = 0,
    Bu32 = 1
}

#[repr(i64)]
enum Ei64 {
    Ai64 = 0,
    Bi64 = 1
}

#[repr(u64)]
enum Eu64 {
    Au64 = 0,
    Bu64 = 1
}

#[repr(isize)]
enum Eint {
    Aint = 0,
    Bint = 1
}

#[repr(usize)]
enum Euint {
    Auint = 0,
    Buint = 1
}

#[repr(u8)]
enum Eu8NonCLike<T> {
    _None,
    _Some(T),
}

#[repr(i64)]
enum Ei64NonCLike<T> {
    _None,
    _Some(T),
}

#[repr(u64)]
enum Eu64NonCLike<T> {
    _None,
    _Some(T),
}

pub fn main() {
    assert_eq!(size_of::<Ei8>(), 1);
    assert_eq!(size_of::<Eu8>(), 1);
    assert_eq!(size_of::<Ei16>(), 2);
    assert_eq!(size_of::<Eu16>(), 2);
    assert_eq!(size_of::<Ei32>(), 4);
    assert_eq!(size_of::<Eu32>(), 4);
    assert_eq!(size_of::<Ei64>(), 8);
    assert_eq!(size_of::<Eu64>(), 8);
    assert_eq!(size_of::<Eint>(), size_of::<isize>());
    assert_eq!(size_of::<Euint>(), size_of::<usize>());
    assert_eq!(size_of::<Eu8NonCLike<()>>(), 1);
    assert_eq!(size_of::<Ei64NonCLike<()>>(), 8);
    assert_eq!(size_of::<Eu64NonCLike<()>>(), 8);
    let u8_expected_size = round_up(9, align_of::<Eu64NonCLike<u8>>());
    assert_eq!(size_of::<Eu64NonCLike<u8>>(), u8_expected_size);
    let array_expected_size = round_up(28, align_of::<Eu64NonCLike<[u32; 5]>>());
    assert_eq!(size_of::<Eu64NonCLike<[u32; 5]>>(), array_expected_size);
    assert_eq!(size_of::<Eu64NonCLike<[u32; 6]>>(), 32);

    assert_eq!(align_of::<Eu32>(), align_of::<u32>());
    assert_eq!(align_of::<Eu64NonCLike<u8>>(), align_of::<u64>());
}

// Rounds x up to the next multiple of a
fn round_up(x: usize, a: usize) -> usize {
    ((x + (a - 1)) / a) * a
}
