use std::cmp::Ordering::{Equal, Less, Greater};

#[test]
fn test_clone() {
    let a = (1, "2");
    let b = a.clone();
    assert_eq!(a, b);
}

#[test]
fn test_tuple_cmp() {
    let (small, big) = ((1, 2, 3), (3, 2, 1));

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
    let s = format!("{:?}", (1,));
    assert_eq!(s, "(1,)");
    let s = format!("{:?}", (1, true));
    assert_eq!(s, "(1, true)");
    let s = format!("{:?}", (1, "hi", true));
    assert_eq!(s, "(1, \"hi\", true)");
}
