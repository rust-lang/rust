use super::*;

#[test]
fn align_constants() {
    assert_eq!(Align::ONE, Align::from_bytes(1).unwrap());
    assert_eq!(Align::EIGHT, Align::from_bytes(8).unwrap());
}

#[test]
#[should_panic(expected = "Value 299 is too big for Size(1 bytes)")]
fn wrapping_range_smallest_range_containing_size_mismatch() {
    WrappingRange::smallest_range_containing(200..300, Size::from_bytes(1));
}

#[test]
fn wrapping_range_smallest_range_containing() {
    #[track_caller]
    fn check(x: impl IntoIterator<Item = u128>, bytes: u64, start: u128, end: u128) {
        assert_eq!(
            WrappingRange::smallest_range_containing(x, Size::from_bytes(bytes)),
            Some(WrappingRange { start, end }),
        );
    }

    assert_eq!(WrappingRange::smallest_range_containing([], Size::from_bytes(1)), None);

    check([7], 1, 7, 7);
    check([7, 7, 7], 1, 7, 7);

    check(0..=127, 1, 0, 127);
    check((0..=127).chain([255]), 1, 255, 127);

    check((-100..=100_i128).map(i128::cast_unsigned), 16, (-100_i128).cast_unsigned(), 100);

    // A wraparound case that's not just "sort them as signed"
    check([10, 100, 160, 220], 1, 100, 10);

    check([0, 0xFF], 1, 0xFF, 0);
    check([0, 0xFF], 2, 0, 0xFF);
    check([0, 0xFFFF], 2, 0xFFFF, 0);
    check([0, 0xFFFF], 4, 0, 0xFFFF);
    check([0, 0xFFFFFFFF], 4, 0xFFFFFFFF, 0);
    check([0, 0xFFFFFFFF], 8, 0, 0xFFFFFFFF);

    check([100, 200], 1, 100, 200);
    check([100, 200, 50], 1, 50, 200);
    check([100, 200, 250], 1, 100, 250);
    check([100, 200, 250, 50], 1, 200, 100);

    check([200, 50], 1, 200, 50);
    check([200, 50, 190], 1, 190, 50);
    check([200, 50, 60], 1, 200, 60);
    check([200, 50, 125], 1, 50, 200);

    // A wraparound range is only potentially interesting when
    // the non-wraparound range covers over half the range.
    check([0, 127], 1, 0, 127); // `0..=127` is smaller
    check([0, 128], 1, 0, 128); // both are the same size
    check([0, 129], 1, 129, 0); // `(..=0) | (129..)` is smaller

    // The mem::Alignment case
    check((0..64).map(|n| 1 << n), 8, 1, i64::MIN.cast_unsigned().into());

    // Both `100..=228` and `..=228 | 100..` are the same size, but we pick the one without zero.
    check([100, 228], 1, 100, 228);

    // The wraparound one here is slightly smaller, so we pick it despite including zero.
    // (The distance 10→96 is 86, compared to 85 for 96→181 and 181→10.)
    check([10, 96, 181], 1, 96, 10);

    // These 4 values are evenly spaced so all 4 candidate ranges have length 193:
    // `(..=32) | (96..)`, `(..=96) | (160..)`, `(..=160) | (224..)`, and `32..=224`.
    // We pick the last one as the only one that doesn't contain zero.
    check([0xA0, 0xE0, 0x20, 0x60], 1, 0x20, 0xE0);

    // Wraparound can still be needed even in `max - min < values.len()`.
    check(std::iter::chain([50; 100], [200; 100]), 1, 200, 50);
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
