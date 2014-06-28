// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#[test]
fn test_clone() {
    let a = (1i, "2");
    let b = a.clone();
    assert_eq!(a, b);
}

#[test]
fn test_getters() {
    macro_rules! test_getter(
        ($x:expr, $valN:ident, $refN:ident, $mutN:ident,
         $init:expr, $incr:expr, $result:expr) => ({
            assert_eq!($x.$valN(), $init);
            assert_eq!(*$x.$refN(), $init);
            *$x.$mutN() += $incr;
            assert_eq!(*$x.$refN(), $result);
        })
    )
    let mut x = (0u8, 1u16, 2u32, 3u64, 4u, 5i8, 6i16, 7i32, 8i64, 9i, 10f32, 11f64);
    test_getter!(x, val0,  ref0,  mut0,  0,    1,   1);
    test_getter!(x, val1,  ref1,  mut1,  1,    1,   2);
    test_getter!(x, val2,  ref2,  mut2,  2,    1,   3);
    test_getter!(x, val3,  ref3,  mut3,  3,    1,   4);
    test_getter!(x, val4,  ref4,  mut4,  4,    1,   5);
    test_getter!(x, val5,  ref5,  mut5,  5,    1,   6);
    test_getter!(x, val6,  ref6,  mut6,  6,    1,   7);
    test_getter!(x, val7,  ref7,  mut7,  7,    1,   8);
    test_getter!(x, val8,  ref8,  mut8,  8,    1,   9);
    test_getter!(x, val9,  ref9,  mut9,  9,    1,   10);
    test_getter!(x, val10, ref10, mut10, 10.0, 1.0, 11.0);
    test_getter!(x, val11, ref11, mut11, 11.0, 1.0, 12.0);
}

#[test]
fn test_tuple_cmp() {
    let (small, big) = ((1u, 2u, 3u), (3u, 2u, 1u));

    let nan = 0.0f64/0.0;

    // PartialEq
    assert_eq!(small, small);
    assert_eq!(big, big);
    assert!(small != big);
    assert!(big != small);

    // PartialOrd
    assert!(small < big);
    assert!(!(small < small));
    assert!(!(big < small));
    assert!(!(big < big));

    assert!(small <= small);
    assert!(big <= big);

    assert!(big > small);
    assert!(small >= small);
    assert!(big >= small);
    assert!(big >= big);

    assert!(!((1.0f64, 2.0f64) < (nan, 3.0)));
    assert!(!((1.0f64, 2.0f64) <= (nan, 3.0)));
    assert!(!((1.0f64, 2.0f64) > (nan, 3.0)));
    assert!(!((1.0f64, 2.0f64) >= (nan, 3.0)));
    assert!(((1.0f64, 2.0f64) < (2.0, nan)));
    assert!(!((2.0f64, 2.0f64) < (2.0, nan)));

    // Ord
    assert!(small.cmp(&small) == Equal);
    assert!(big.cmp(&big) == Equal);
    assert!(small.cmp(&big) == Less);
    assert!(big.cmp(&small) == Greater);
}

#[test]
fn test_show() {
    let s = format!("{}", (1i,));
    assert_eq!(s.as_slice(), "(1,)");
    let s = format!("{}", (1i, true));
    assert_eq!(s.as_slice(), "(1, true)");
    let s = format!("{}", (1i, "hi", true));
    assert_eq!(s.as_slice(), "(1, hi, true)");
}
