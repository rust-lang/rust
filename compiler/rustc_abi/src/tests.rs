use super::*;

#[test]
fn align_constants() {
    assert_eq!(Align::ONE, Align::from_bytes(1).unwrap());
    assert_eq!(Align::EIGHT, Align::from_bytes(8).unwrap());
}

#[test]
fn wrapping_range_contains_range() {
    let size16 = Size::from_bytes(16);

    let a = WrappingRange { start: 10, end: 20 };
    assert!(a.contains_range(a, size16));
    assert!(a.contains_range(WrappingRange { start: 11, end: 19 }, size16));
    assert!(a.contains_range(WrappingRange { start: 10, end: 10 }, size16));
    assert!(a.contains_range(WrappingRange { start: 20, end: 20 }, size16));
    assert!(!a.contains_range(WrappingRange { start: 10, end: 21 }, size16));
    assert!(!a.contains_range(WrappingRange { start: 9, end: 20 }, size16));
    assert!(!a.contains_range(WrappingRange { start: 4, end: 6 }, size16));
    assert!(!a.contains_range(WrappingRange { start: 24, end: 26 }, size16));

    assert!(!a.contains_range(WrappingRange { start: 16, end: 14 }, size16));

    let b = WrappingRange { start: 20, end: 10 };
    assert!(b.contains_range(b, size16));
    assert!(b.contains_range(WrappingRange { start: 20, end: 20 }, size16));
    assert!(b.contains_range(WrappingRange { start: 10, end: 10 }, size16));
    assert!(b.contains_range(WrappingRange { start: 0, end: 10 }, size16));
    assert!(b.contains_range(WrappingRange { start: 20, end: 30 }, size16));
    assert!(b.contains_range(WrappingRange { start: 20, end: 9 }, size16));
    assert!(b.contains_range(WrappingRange { start: 21, end: 10 }, size16));
    assert!(b.contains_range(WrappingRange { start: 999, end: 9999 }, size16));
    assert!(b.contains_range(WrappingRange { start: 999, end: 9 }, size16));
    assert!(!b.contains_range(WrappingRange { start: 19, end: 19 }, size16));
    assert!(!b.contains_range(WrappingRange { start: 11, end: 11 }, size16));
    assert!(!b.contains_range(WrappingRange { start: 19, end: 11 }, size16));
    assert!(!b.contains_range(WrappingRange { start: 11, end: 19 }, size16));

    let f = WrappingRange { start: 0, end: u128::MAX };
    assert!(f.contains_range(WrappingRange { start: 10, end: 20 }, size16));
    assert!(f.contains_range(WrappingRange { start: 20, end: 10 }, size16));

    let g = WrappingRange { start: 2, end: 1 };
    assert!(g.contains_range(WrappingRange { start: 10, end: 20 }, size16));
    assert!(g.contains_range(WrappingRange { start: 20, end: 10 }, size16));

    let size1 = Size::from_bytes(1);
    let u8r = WrappingRange { start: 0, end: 255 };
    let i8r = WrappingRange { start: 128, end: 127 };
    assert!(u8r.contains_range(i8r, size1));
    assert!(i8r.contains_range(u8r, size1));
    assert!(!u8r.contains_range(i8r, size16));
    assert!(i8r.contains_range(u8r, size16));

    let boolr = WrappingRange { start: 0, end: 1 };
    assert!(u8r.contains_range(boolr, size1));
    assert!(i8r.contains_range(boolr, size1));
    assert!(!boolr.contains_range(u8r, size1));
    assert!(!boolr.contains_range(i8r, size1));

    let cmpr = WrappingRange { start: 255, end: 1 };
    assert!(u8r.contains_range(cmpr, size1));
    assert!(i8r.contains_range(cmpr, size1));
    assert!(!cmpr.contains_range(u8r, size1));
    assert!(!cmpr.contains_range(i8r, size1));

    assert!(!boolr.contains_range(cmpr, size1));
    assert!(cmpr.contains_range(boolr, size1));
}
