use {std::ops, text_size::*};

fn size(x: u32) -> TextSize {
    TextSize::from(x)
}

fn range(x: ops::Range<u32>) -> TextRange {
    TextRange::from(x)
}

#[test]
fn sum() {
    let xs: Vec<TextSize> = vec![size(0), size(1), size(2)];
    assert_eq!(xs.iter().copied().sum::<TextSize>(), size(3));
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
    assert_eq!(TextSize::MAX.checked_add(size(1)), None);
}

#[test]
#[rustfmt::skip]
fn contains() {
    assert!(   range(2..4).contains(range(2..3)));
    assert!( ! range(2..4).contains(range(1..3)));
}

#[test]
fn intersection() {
    assert_eq!(
        TextRange::intersection(range(1..2), range(2..3)),
        Some(range(2..2))
    );
    assert_eq!(
        TextRange::intersection(range(1..5), range(2..3)),
        Some(range(2..3))
    );
    assert_eq!(TextRange::intersection(range(1..2), range(3..4)), None);
}

#[test]
fn covering() {
    assert_eq!(TextRange::covering(range(1..2), range(2..3)), range(1..3));
    assert_eq!(TextRange::covering(range(1..5), range(2..3)), range(1..5));
    assert_eq!(TextRange::covering(range(1..2), range(4..5)), range(1..5));
}

#[test]
#[rustfmt::skip]
fn contains_point() {
    assert!( ! range(1..3).contains_exclusive(size(0)));
    assert!(   range(1..3).contains_exclusive(size(1)));
    assert!(   range(1..3).contains_exclusive(size(2)));
    assert!( ! range(1..3).contains_exclusive(size(3)));
    assert!( ! range(1..3).contains_exclusive(size(4)));

    assert!( ! range(1..3).contains_inclusive(size(0)));
    assert!(   range(1..3).contains_inclusive(size(1)));
    assert!(   range(1..3).contains_inclusive(size(2)));
    assert!(   range(1..3).contains_inclusive(size(3)));
    assert!( ! range(1..3).contains_inclusive(size(4)));
}
