use {std::ops, text_size::*};

fn size(x: u32) -> TextSize {
    TextSize::from(x)
}

fn range(x: ops::Range<u32>) -> TextRange {
    TextRange::new(x.start.into(), x.end.into())
}

#[test]
fn sum() {
    let xs: Vec<TextSize> = vec![size(0), size(1), size(2)];
    assert_eq!(xs.iter().sum::<TextSize>(), size(3));
    assert_eq!(xs.into_iter().sum::<TextSize>(), size(3));
}

#[test]
fn math() {
    assert_eq!(size(10) + size(5), size(15));
    assert_eq!(size(10) - size(5), size(5));
}

#[test]
fn checked_math() {
    assert_eq!(size(1).checked_add(size(1)), Some(size(2)));
    assert_eq!(size(1).checked_sub(size(1)), Some(size(0)));
    assert_eq!(size(1).checked_sub(size(2)), None);
    assert_eq!(size(!0).checked_add(size(1)), None);
}

#[test]
#[rustfmt::skip]
fn contains() {
    assert!(   range(2..4).contains_range(range(2..3)));
    assert!( ! range(2..4).contains_range(range(1..3)));
}

#[test]
fn intersect() {
    assert_eq!(range(1..2).intersect(range(2..3)), Some(range(2..2)));
    assert_eq!(range(1..5).intersect(range(2..3)), Some(range(2..3)));
    assert_eq!(range(1..2).intersect(range(3..4)), None);
}

#[test]
fn cover() {
    assert_eq!(range(1..2).cover(range(2..3)), range(1..3));
    assert_eq!(range(1..5).cover(range(2..3)), range(1..5));
    assert_eq!(range(1..2).cover(range(4..5)), range(1..5));
}

#[test]
fn cover_offset() {
    assert_eq!(range(1..3).cover_offset(size(0)), range(0..3));
    assert_eq!(range(1..3).cover_offset(size(1)), range(1..3));
    assert_eq!(range(1..3).cover_offset(size(2)), range(1..3));
    assert_eq!(range(1..3).cover_offset(size(3)), range(1..3));
    assert_eq!(range(1..3).cover_offset(size(4)), range(1..4));
}

#[test]
#[rustfmt::skip]
fn contains_point() {
    assert!( ! range(1..3).contains(size(0)));
    assert!(   range(1..3).contains(size(1)));
    assert!(   range(1..3).contains(size(2)));
    assert!( ! range(1..3).contains(size(3)));
    assert!( ! range(1..3).contains(size(4)));

    assert!( ! range(1..3).contains_inclusive(size(0)));
    assert!(   range(1..3).contains_inclusive(size(1)));
    assert!(   range(1..3).contains_inclusive(size(2)));
    assert!(   range(1..3).contains_inclusive(size(3)));
    assert!( ! range(1..3).contains_inclusive(size(4)));
}
