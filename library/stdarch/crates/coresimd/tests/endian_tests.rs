#![feature(stdsimd)]
#![cfg_attr(stdsimd_strict, deny(warnings))]

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
    #[cfg_attr(rustfmt, rustfmt_skip)]
    let x = i8x16::new(
        0, 1, 2, 3, 4, 5, 6, 7,
        8, 9, 10, 11, 12, 13, 14, 15,
    );
    let t: i16x8 = unsafe { mem::transmute(x) };
    let e: i16x8 = if cfg!(target_endian = "little") {
        i16x8::new(256, 770, 1284, 1798, 2312, 2826, 3340, 3854)
    } else {
        i16x8::new(1, 515, 1029, 1543, 2057, 2571, 3085, 3599)
    };
    assert_eq!(t, e);
}

#[test]
fn endian_casts() {
    #[cfg_attr(rustfmt, rustfmt_skip)]
    let x = i8x16::new(
        0, 1, 2, 3, 4, 5, 6, 7,
        8, 9, 10, 11, 12, 13, 14, 15,
    );
    let t: i16x16 = x.into(); // simd_cast
    #[cfg_attr(rustfmt, rustfmt_skip)]
    let e = i16x16::new(
        0, 1, 2, 3, 4, 5, 6, 7,
        8, 9, 10, 11, 12, 13, 14, 15,
    );
    assert_eq!(t, e);
}

#[test]
fn endian_load_and_stores() {
    #[cfg_attr(rustfmt, rustfmt_skip)]
    let x = i8x16::new(
        0, 1, 2, 3, 4, 5, 6, 7,
        8, 9, 10, 11, 12, 13, 14, 15,
    );
    let mut y: [i16; 8] = [0; 8];
    x.store_unaligned(unsafe {
        slice::from_raw_parts_mut(&mut y as *mut _ as *mut i8, 16)
    });

    let e: [i16; 8] = if cfg!(target_endian = "little") {
        [256, 770, 1284, 1798, 2312, 2826, 3340, 3854]
    } else {
        [1, 515, 1029, 1543, 2057, 2571, 3085, 3599]
    };
    assert_eq!(y, e);

    let z = i8x16::load_unaligned(unsafe {
        slice::from_raw_parts(&y as *const _ as *const i8, 16)
    });
    assert_eq!(z, x);
}

#[test]
fn endian_array_union() {
    union A {
        data: [f32; 4],
        vec: f32x4,
    }
    let x: [f32; 4] = unsafe {
        A {
            vec: f32x4::new(0., 1., 2., 3.),
        }.data
    };
    assert_eq!(x[0], 0_f32);
    assert_eq!(x[1], 1_f32);
    assert_eq!(x[2], 2_f32);
    assert_eq!(x[3], 3_f32);
    let y: f32x4 = unsafe {
        A {
            data: [3., 2., 1., 0.],
        }.vec
    };
    assert_eq!(y, f32x4::new(3., 2., 1., 0.));

    union B {
        data: [i8; 16],
        vec: i8x16,
    }
    #[cfg_attr(rustfmt, rustfmt_skip)]
    let x = i8x16::new(
        0, 1, 2, 3, 4, 5, 6, 7,
        8, 9, 10, 11, 12, 13, 14, 15,
    );
    let x: [i8; 16] = unsafe { B { vec: x }.data };

    for i in 0..16 {
        assert_eq!(x[i], i as i8);
    }

    #[cfg_attr(rustfmt, rustfmt_skip)]
    let y = [
        15, 14, 13, 12, 11, 19, 9, 8,
        7, 6, 5, 4, 3, 2, 1, 0
    ];
    #[cfg_attr(rustfmt, rustfmt_skip)]
    let e = i8x16::new(
        15, 14, 13, 12, 11, 19, 9, 8,
        7, 6, 5, 4, 3, 2, 1, 0
    );
    let z = unsafe { B { data: y }.vec };
    assert_eq!(z, e);

    union C {
        data: [i16; 8],
        vec: i8x16,
    }
    #[cfg_attr(rustfmt, rustfmt_skip)]
    let x = i8x16::new(
        0, 1, 2, 3, 4, 5, 6, 7,
        8, 9, 10, 11, 12, 13, 14, 15,
    );
    let x: [i16; 8] = unsafe { C { vec: x }.data };

    let e: [i16; 8] = if cfg!(target_endian = "little") {
        [256, 770, 1284, 1798, 2312, 2826, 3340, 3854]
    } else {
        [1, 515, 1029, 1543, 2057, 2571, 3085, 3599]
    };
    assert_eq!(x, e);
}

#[test]
fn endian_tuple_access() {
    type F32x4T = (f32, f32, f32, f32);
    union A {
        data: F32x4T,
        vec: f32x4,
    }
    let x: F32x4T = unsafe {
        A {
            vec: f32x4::new(0., 1., 2., 3.),
        }.data
    };
    assert_eq!(x.0, 0_f32);
    assert_eq!(x.1, 1_f32);
    assert_eq!(x.2, 2_f32);
    assert_eq!(x.3, 3_f32);
    let y: f32x4 = unsafe {
        A {
            data: (3., 2., 1., 0.),
        }.vec
    };
    assert_eq!(y, f32x4::new(3., 2., 1., 0.));

    #[cfg_attr(rustfmt, rustfmt_skip)]
    type I8x16T = (i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8);
    union B {
        data: I8x16T,
        vec: i8x16,
    }

    #[cfg_attr(rustfmt, rustfmt_skip)]
    let x = i8x16::new(
        0, 1, 2, 3, 4, 5, 6, 7,
        8, 9, 10, 11, 12, 13, 14, 15,
    );
    let x: I8x16T = unsafe { B { vec: x }.data };

    assert_eq!(x.0, 0);
    assert_eq!(x.1, 1);
    assert_eq!(x.2, 2);
    assert_eq!(x.3, 3);
    assert_eq!(x.4, 4);
    assert_eq!(x.5, 5);
    assert_eq!(x.6, 6);
    assert_eq!(x.7, 7);
    assert_eq!(x.8, 8);
    assert_eq!(x.9, 9);
    assert_eq!(x.10, 10);
    assert_eq!(x.11, 11);
    assert_eq!(x.12, 12);
    assert_eq!(x.13, 13);
    assert_eq!(x.14, 14);
    assert_eq!(x.15, 15);

    #[cfg_attr(rustfmt, rustfmt_skip)]
    let y = (
        15, 14, 13, 12, 11, 10, 9, 8,
        7, 6, 5, 4, 3, 2, 1, 0
    );
    let z: i8x16 = unsafe { B { data: y }.vec };
    #[cfg_attr(rustfmt, rustfmt_skip)]
    let e = i8x16::new(
        15, 14, 13, 12, 11, 10, 9, 8,
        7, 6, 5, 4, 3, 2, 1, 0
    );
    assert_eq!(e, z);

    #[cfg_attr(rustfmt, rustfmt_skip)]
    type I16x8T = (i16, i16, i16, i16, i16, i16, i16, i16);
    union C {
        data: I16x8T,
        vec: i8x16,
    }

    #[cfg_attr(rustfmt, rustfmt_skip)]
    let x = i8x16::new(
        0, 1, 2, 3, 4, 5, 6, 7,
        8, 9, 10, 11, 12, 13, 14, 15,
    );
    let x: I16x8T = unsafe { C { vec: x }.data };

    let e: [i16; 8] = if cfg!(target_endian = "little") {
        [256, 770, 1284, 1798, 2312, 2826, 3340, 3854]
    } else {
        [1, 515, 1029, 1543, 2057, 2571, 3085, 3599]
    };
    assert_eq!(x.0, e[0]);
    assert_eq!(x.1, e[1]);
    assert_eq!(x.2, e[2]);
    assert_eq!(x.3, e[3]);
    assert_eq!(x.4, e[4]);
    assert_eq!(x.5, e[5]);
    assert_eq!(x.6, e[6]);
    assert_eq!(x.7, e[7]);

    // Without repr(C) this produces total garbage.
    // FIXME: investigate more, this is maybe due to
    // to tuple field reordering to minimize padding.
    #[cfg_attr(rustfmt, rustfmt_skip)]
    #[repr(C)]
    #[derive(Copy ,Clone)]
    pub struct Tup(pub i8, pub i8, pub i16, pub i8, pub i8, pub i16,
                   pub i8, pub i8, pub i16, pub i8, pub i8, pub i16);

    union D {
        data: Tup,
        vec: i8x16,
    }

    #[cfg_attr(rustfmt, rustfmt_skip)]
    let x = i8x16::new(
        0, 1, 2, 3, 4, 5, 6, 7,
        8, 9, 10, 11, 12, 13, 14, 15,
    );
    let x: Tup = unsafe { D { vec: x }.data };

    let e: [i16; 12] = if cfg!(target_endian = "little") {
        [0, 1, 770, 4, 5, 1798, 8, 9, 2826, 12, 13, 3854]
    } else {
        [0, 1, 515, 4, 5, 1543, 8, 9, 2571, 12, 13, 3599]
    };
    assert_eq!(x.0 as i16, e[0]);
    assert_eq!(x.1 as i16, e[1]);
    assert_eq!(x.2 as i16, e[2]);
    assert_eq!(x.3 as i16, e[3]);
    assert_eq!(x.4 as i16, e[4]);
    assert_eq!(x.5 as i16, e[5]);
    assert_eq!(x.6 as i16, e[6]);
    assert_eq!(x.7 as i16, e[7]);
    assert_eq!(x.8 as i16, e[8]);
    assert_eq!(x.9 as i16, e[9]);
    assert_eq!(x.10 as i16, e[10]);
    assert_eq!(x.11 as i16, e[11]);
}
