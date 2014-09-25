// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use core::cmp::lexical_ordering;
use core::cmp::{ partial_min, partial_max };

#[test]
fn test_int_totalord() {
    assert_eq!(5i.cmp(&10), Less);
    assert_eq!(10i.cmp(&5), Greater);
    assert_eq!(5i.cmp(&5), Equal);
    assert_eq!((-5i).cmp(&12), Less);
    assert_eq!(12i.cmp(&-5), Greater);
}

#[test]
fn test_mut_int_totalord() {
    assert_eq!((&mut 5i).cmp(&&mut 10), Less);
    assert_eq!((&mut 10i).cmp(&&mut 5), Greater);
    assert_eq!((&mut 5i).cmp(&&mut 5), Equal);
    assert_eq!((&mut -5i).cmp(&&mut 12), Less);
    assert_eq!((&mut 12i).cmp(&&mut -5), Greater);
}

#[test]
fn test_ordering_reverse() {
    assert_eq!(Less.reverse(), Greater);
    assert_eq!(Equal.reverse(), Equal);
    assert_eq!(Greater.reverse(), Less);
}

#[test]
fn test_ordering_order() {
    assert!(Less < Equal);
    assert_eq!(Greater.cmp(&Less), Greater);
}

#[test]
#[allow(deprecated)]
fn test_lexical_ordering() {
    fn t(o1: Ordering, o2: Ordering, e: Ordering) {
        assert_eq!(lexical_ordering(o1, o2), e);
    }

    let xs = [Less, Equal, Greater];
    for &o in xs.iter() {
        t(Less, o, Less);
        t(Equal, o, o);
        t(Greater, o, Greater);
     }
}

#[test]
fn test_partial_min() {
    use core::f64::NAN;
    let data_integer = [
        // a, b, result
        (0i, 0i, Some(0i)),
        (1i, 0i, Some(0i)),
        (0i, 1i, Some(0i)),
        (-1i, 0i, Some(-1i)),
        (0i, -1i, Some(-1i))
    ];

    let data_float = [
        // a, b, result
        (0.0f64, 0.0f64, Some(0.0f64)),
        (1.0f64, 0.0f64, Some(0.0f64)),
        (0.0f64, 1.0f64, Some(0.0f64)),
        (-1.0f64, 0.0f64, Some(-1.0f64)),
        (0.0f64, -1.0f64, Some(-1.0f64)),
        (NAN, NAN, None),
        (NAN, 1.0f64, None),
        (1.0f64, NAN, None)
    ];

    for &(a, b, result) in data_integer.iter() {
        assert!(partial_min(a, b) == result);
    }

    for &(a, b, result) in data_float.iter() {
        assert!(partial_min(a, b) == result);
    }
}

#[test]
fn test_partial_max() {
    use core::f64::NAN;
    let data_integer = [
        // a, b, result
        (0i, 0i, Some(0i)),
        (1i, 0i, Some(1i)),
        (0i, 1i, Some(1i)),
        (-1i, 0i, Some(0i)),
        (0i, -1i, Some(0i))
    ];

    let data_float = [
        // a, b, result
        (0.0f64, 0.0f64, Some(0.0f64)),
        (1.0f64, 0.0f64, Some(1.0f64)),
        (0.0f64, 1.0f64, Some(1.0f64)),
        (-1.0f64, 0.0f64, Some(0.0f64)),
        (0.0f64, -1.0f64, Some(0.0f64)),
        (NAN, NAN, None),
        (NAN, 1.0f64, None),
        (1.0f64, NAN, None)
    ];

    for &(a, b, result) in data_integer.iter() {
        assert!(partial_max(a, b) == result);
    }

    for &(a, b, result) in data_float.iter() {
        assert!(partial_max(a, b) == result);
    }
}

#[test]
fn test_user_defined_eq() {
    // Our type.
    struct SketchyNum {
        num : int
    }

    // Our implementation of `PartialEq` to support `==` and `!=`.
    impl PartialEq for SketchyNum {
        // Our custom eq allows numbers which are near each other to be equal! :D
        fn eq(&self, other: &SketchyNum) -> bool {
            (self.num - other.num).abs() < 5
        }
    }

    // Now these binary operators will work when applied!
    assert!(SketchyNum {num: 37} == SketchyNum {num: 34});
    assert!(SketchyNum {num: 25} != SketchyNum {num: 57});
}
