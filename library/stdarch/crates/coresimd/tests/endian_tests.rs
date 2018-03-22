#![feature(cfg_target_feature, stdsimd)]
#![cfg_attr(feature = "strict", deny(warnings))]

extern crate core;
extern crate coresimd;

use core::{mem, slice};
use coresimd::simd::*;

#[test]
fn endian_indexing() {
    let v = i32x4::new(0, 1, 2, 3);
    assert_eq!(v.extract(0), 0);
    assert_eq!(v.extract(1), 1);
    assert_eq!(v.extract(2), 2);
    assert_eq!(v.extract(3), 3);
}

#[test]
fn endian_bitcasts() {
    let x = i8x16::new(
        0,
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        9,
        10,
        11,
        12,
        13,
        14,
        15,
    );
    let t: i16x8 = unsafe { mem::transmute(x) };
    if cfg!(target_endian = "little") {
        let t_el = i16x8::new(256, 770, 1284, 1798, 2312, 2826, 3340, 3854);
        assert_eq!(t, t_el);
    } else if cfg!(target_endian = "big") {
        let t_be = i16x8::new(1, 515, 1029, 1543, 2057, 2571, 3085, 3599);
        assert_eq!(t, t_be);
    }
}

#[test]
fn endian_casts() {
    let x = i8x16::new(
        0,
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        9,
        10,
        11,
        12,
        13,
        14,
        15,
    );
    let t: i16x16 = x.into(); // simd_cast
    let e = i16x16::new(
        0,
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        9,
        10,
        11,
        12,
        13,
        14,
        15,
    );
    assert_eq!(t, e);
}

#[test]
fn endian_load_and_stores() {
    let x = i8x16::new(
        0,
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        9,
        10,
        11,
        12,
        13,
        14,
        15,
    );
    let mut y: [i16; 8] = [0; 8];
    x.store_unaligned(unsafe {
        slice::from_raw_parts_mut(&mut y as *mut _ as *mut i8, 16)
    });

    if cfg!(target_endian = "little") {
        let e: [i16; 8] = [256, 770, 1284, 1798, 2312, 2826, 3340, 3854];
        assert_eq!(y, e);
    } else if cfg!(target_endian = "big") {
        let e: [i16; 8] = [1, 515, 1029, 1543, 2057, 2571, 3085, 3599];
        assert_eq!(y, e);
    }

    let z = i8x16::load_unaligned(unsafe {
        slice::from_raw_parts(&y as *const _ as *const i8, 16)
    });
    assert_eq!(z, x);
}
