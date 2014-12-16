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
        ($val:expr, $init:expr plus $incr:expr makes $result:expr) => ({
            assert_eq!($val, $init);
            assert_eq!(*&$val, $init);
            *&mut $val += $incr;
            assert_eq!(*&$val, $result);
        })
    )

    let mut x = (0u8, 1u16, 2u32, 3u64, 4u, 5i8, 6i16, 7i32, 8i64, 9i, 10f32, 11f64);
    test_getter!(x.0, 0 plus 1 makes 1);
    test_getter!(x.1, 1 plus 1 makes 2);
    test_getter!(x.2, 2 plus 1 makes 3);
    test_getter!(x.3, 3 plus 1 makes 4);
    test_getter!(x.4, 4 plus 1 makes 5);
    test_getter!(x.5, 5 plus 1 makes 6);
    test_getter!(x.6, 6 plus 1 makes 7);
    test_getter!(x.7, 7 plus 1 makes 8);
    test_getter!(x.8, 8 plus 1 makes 9);
    test_getter!(x.9, 9 plus 1 makes 10);
    test_getter!(x.10, 10.0 plus 1.0 makes 11.0);
    test_getter!(x.11, 11.0 plus 1.0 makes 12.0);
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
    assert_eq!(s, "(1,)");
    let s = format!("{}", (1i, true));
    assert_eq!(s, "(1, true)");
    let s = format!("{}", (1i, "hi", true));
    assert_eq!(s, "(1, hi, true)");
}
