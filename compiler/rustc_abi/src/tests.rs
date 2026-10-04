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

#[test]
fn wrapping_range_count_unused() {
    let zero = WrappingRange { start: 0, end: 0 };
    assert_eq!(zero.count_unused(Size::from_bytes(1)), u8::MAX.into());
    assert_eq!(zero.count_unused(Size::from_bytes(2)), u16::MAX.into());

    let full = WrappingRange { start: 2, end: 1 };
    assert_eq!(full.count_unused(Size::from_bytes(1)), 0);
    assert_eq!(full.count_unused(Size::from_bytes(2)), 0);

    let byte = WrappingRange::full(Size::from_bytes(1));
    assert_eq!(byte.count_unused(Size::from_bytes(1)), 0);
    assert_eq!(byte.count_unused(Size::from_bytes(2)), 0x10000 - 0x100);
}

fn niche_16(start: u128, end: u128) -> Niche {
    Niche {
        offset: Size::from_bytes(123),
        value: Primitive::Int(Integer::I16, true),
        valid_range: WrappingRange { start, end },
    }
}

#[test]
fn niche_reserve_insufficient_space() {
    let n = niche_16(1, u16::MAX.into());
    assert_eq!(n.reserve(&FailCx, 2), None);

    // Callers don't do any pre-checks, so can show up with type-impossible requests too.
    // For example, layout asks if it can store 1071 values in a byte for
    // `cranelift_assembler_x64::inst::Inst<isa::x64::inst::external::CraneliftRegisters>`
    let n = niche_16(1, 15);
    assert_eq!(n.reserve(&FailCx, u64::MAX.into()), None);
}

#[test]
fn niche_reserve_full_ranges() {
    let full_16 = WrappingRange::full(Size::from_bits(16));

    let n = niche_16(1, u16::MAX.into());
    let (first, scalar) = n.reserve(&FailCx, 1).unwrap();
    assert_eq!(first, 0);
    assert_eq!(scalar.valid_range(&FailCx), full_16);

    let n = niche_16(0, (u16::MAX - 1).into());
    let (first, scalar) = n.reserve(&FailCx, 1).unwrap();
    assert_eq!(first, u16::MAX.into());
    assert_eq!(scalar.valid_range(&FailCx), full_16);

    let n = niche_16(5, 3);
    let (first, scalar) = n.reserve(&FailCx, 1).unwrap();
    assert_eq!(first, 4);
    // It would also be fine for this to be `start: 5, end: 4`, but this is what we do now.
    assert_eq!(scalar.valid_range(&FailCx), WrappingRange { start: 4, end: 3 });
}

#[test]
fn niche_reserve_zero_adjacent() {
    let n = niche_16(0, 0);
    let (first, scalar) = n.reserve(&FailCx, 1).unwrap();
    assert_eq!(first, 1);
    assert_eq!(scalar.valid_range(&FailCx), WrappingRange { start: 0, end: 1 });
    let (first, scalar) = n.reserve(&FailCx, 11).unwrap();
    assert_eq!(first, 1);
    assert_eq!(scalar.valid_range(&FailCx), WrappingRange { start: 0, end: 11 });

    let n = niche_16(0, 0x7FFF);
    let (first, scalar) = n.reserve(&FailCx, 1).unwrap();
    assert_eq!(first, 0xFFFF);
    assert_eq!(scalar.valid_range(&FailCx), WrappingRange { start: 0xFFFF, end: 0x7FFF });
    let (first, scalar) = n.reserve(&FailCx, 16).unwrap();
    assert_eq!(first, 0xFFF0);
    assert_eq!(scalar.valid_range(&FailCx), WrappingRange { start: 0xFFF0, end: 0x7FFF });

    let n = niche_16(0x8000, 0);
    let (first, scalar) = n.reserve(&FailCx, 1).unwrap();
    assert_eq!(first, 1);
    assert_eq!(scalar.valid_range(&FailCx), WrappingRange { start: 0x8000, end: 1 });
    let (first, scalar) = n.reserve(&FailCx, 16).unwrap();
    assert_eq!(first, 1);
    assert_eq!(scalar.valid_range(&FailCx), WrappingRange { start: 0x8000, end: 16 });
}

#[test]
fn niche_reserve_multiple_no_wraparound() {
    let n = niche_16(10, 1000);
    let (first, scalar) = n.reserve(&FailCx, 9).unwrap();
    assert_eq!(first, 1);
    assert_eq!(scalar.valid_range(&FailCx), WrappingRange { start: 1, end: 1000 });
    let (first, scalar) = n.reserve(&FailCx, 10).unwrap();
    assert_eq!(first, 0);
    assert_eq!(scalar.valid_range(&FailCx), WrappingRange { start: 0, end: 1000 });

    let n = niche_16(10, 40_000);
    let (first, scalar) = n.reserve(&FailCx, 9).unwrap();
    assert_eq!(first, 1);
    assert_eq!(scalar.valid_range(&FailCx), WrappingRange { start: 1, end: 40_000 });
    let (first, scalar) = n.reserve(&FailCx, 10).unwrap();
    assert_eq!(first, 40_001);
    assert_eq!(scalar.valid_range(&FailCx), WrappingRange { start: 10, end: 40_010 });

    let n = niche_16(40_000, 0xFFFF - 9);
    let (first, scalar) = n.reserve(&FailCx, 9).unwrap();
    assert_eq!(first, 0xFFFF - 8);
    assert_eq!(scalar.valid_range(&FailCx), WrappingRange { start: 40_000, end: 0xFFFF });
    let (first, scalar) = n.reserve(&FailCx, 10).unwrap();
    assert_eq!(first, 0xFFFF - 8);
    assert_eq!(scalar.valid_range(&FailCx), WrappingRange { start: 40_000, end: 0 });

    let n = niche_16(10_000, 0xFFFF - 9);
    let (first, scalar) = n.reserve(&FailCx, 9).unwrap();
    assert_eq!(first, 0xFFFF - 8);
    assert_eq!(scalar.valid_range(&FailCx), WrappingRange { start: 10_000, end: 0xFFFF });
    let (first, scalar) = n.reserve(&FailCx, 10).unwrap();
    assert_eq!(first, 9_990);
    assert_eq!(scalar.valid_range(&FailCx), WrappingRange { start: 9_990, end: 0xFFFF - 9 });
}

#[test]
fn niche_reserve_smaller_magnitude() {
    let cases: &[([i16; 2], u128, [i16; 2])] = &[
        ([10, 1000], 1, [9, 1000]),
        ([-1000, -10], 1, [-1000, -9]),
        ([-2, 3], 1, [-3, 3]),
        ([-3, 2], 1, [-3, 3]),
        ([-127, 127], 1, [-128, 127]),
        ([-1, 1], 1, [-2, 1]),
        // Already wrapping, with different counts
        ([-2, 1], 1, [-2, 2]),
        ([-2, 1], 2, [-2, 3]),
        // Multiple with forced wraparound
        ([0x0020, -0x0010], 0x22, [0x0020, 0x0012]),
        ([0x0010, -0x0020], 0x22, [-0x0012, -0x0020]),
    ];
    for &([n_start, n_end], count, [x_start, x_end]) in cases {
        let widen = |x: i16| x as u16 as u128;
        let n = niche_16(widen(n_start), widen(n_end));
        let (_, scalar) = n.reserve(&FailCx, count).unwrap();
        assert_eq!(
            scalar.valid_range(&FailCx),
            WrappingRange { start: widen(x_start), end: widen(x_end) },
            "Input niche {n:?}",
        );
    }
}

struct FailCx;
impl HasDataLayout for FailCx {
    fn data_layout(&self) -> &TargetDataLayout {
        unimplemented!()
    }
}
