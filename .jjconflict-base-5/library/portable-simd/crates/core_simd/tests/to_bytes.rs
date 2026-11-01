#![feature(portable_simd)]

use core_simd::simd::{Simd, ToBytes};

#[test]
fn byte_convert() {
    let int = Simd::<u32, 2>::from_array([0xdeadbeef, 0x8badf00d]);
    let ne_bytes = int.to_ne_bytes();
    let be_bytes = int.to_be_bytes();
    let le_bytes = int.to_le_bytes();
    assert_eq!(int[0].to_ne_bytes(), ne_bytes[..4]);
    assert_eq!(int[1].to_ne_bytes(), ne_bytes[4..]);
    assert_eq!(int[0].to_be_bytes(), be_bytes[..4]);
    assert_eq!(int[1].to_be_bytes(), be_bytes[4..]);
    assert_eq!(int[0].to_le_bytes(), le_bytes[..4]);
    assert_eq!(int[1].to_le_bytes(), le_bytes[4..]);
    assert_eq!(Simd::<u32, 2>::from_ne_bytes(ne_bytes), int);
    assert_eq!(Simd::<u32, 2>::from_be_bytes(be_bytes), int);
    assert_eq!(Simd::<u32, 2>::from_le_bytes(le_bytes), int);
}
