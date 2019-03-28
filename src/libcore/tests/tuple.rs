use std::cmp::Ordering::{Equal, Less, Greater};
use std::f64::NAN;

#[test]
fn test_clone() {
    let a = (1, "2");
    let b = a.clone();
    assert_eq!(a, b);
}

#[test]
fn test_partial_eq() {
    let (small, big) = ((1, 2, 3), (3, 2, 1));
    assert_eq!(small, small);
    assert_eq!(big, big);
    assert_ne!(small, big);
    assert_ne!(big, small);
}

#[test]
fn test_partial_ord() {
    let (small, big) = ((1, 2, 3), (3, 2, 1));

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

    assert!(!((1.0f64, 2.0f64) < (NAN, 3.0)));
    assert!(!((1.0f64, 2.0f64) <= (NAN, 3.0)));
    assert!(!((1.0f64, 2.0f64) > (NAN, 3.0)));
    assert!(!((1.0f64, 2.0f64) >= (NAN, 3.0)));
    assert!(((1.0f64, 2.0f64) < (2.0, NAN)));
    assert!(!((2.0f64, 2.0f64) < (2.0, NAN)));
}

#[test]
fn test_ord() {
    let (small, big) = ((1, 2, 3), (3, 2, 1));
    assert_eq!(small.cmp(&small), Equal);
    assert_eq!(big.cmp(&big), Equal);
    assert_eq!(small.cmp(&big), Less);
    assert_eq!(big.cmp(&small), Greater);
}

#[test]
fn test_show() {
    let s = format!("{:?}", (1,));
    assert_eq!(s, "(1,)");
    let s = format!("{:?}", (1, true));
    assert_eq!(s, "(1, true)");
    let s = format!("{:?}", (1, "hi", true));
    assert_eq!(s, "(1, \"hi\", true)");
}
