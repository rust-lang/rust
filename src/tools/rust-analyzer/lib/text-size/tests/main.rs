use text_size::*;

fn r(from: u32, to: u32) -> TextRange {
    TextRange::from(from..to)
}

#[test]
fn sum() {
    let xs: Vec<TextSize> = vec![0.into(), 1.into(), 2.into()];
    assert_eq!(xs.iter().sum::<TextSize>(), 3.into());
    assert_eq!(xs.into_iter().sum::<TextSize>(), 3.into());
}

#[test]
fn math() {
    let range = r(10, 20);
    assert_eq!(range + 5, r(15, 25));
    assert_eq!(range - 5, r(5, 15));
}

#[test]
fn checked_math() {
    let x: TextSize = 1.into();
    assert_eq!(x.checked_sub(1), Some(0.into()));
    assert_eq!(x.checked_sub(2), None);

    assert_eq!(r(1, 2).checked_sub(1), Some(r(0, 1)));
    assert_eq!(x.checked_sub(2), None);
}

#[test]
fn contains() {
    let r1 = r(2, 4);
    let r2 = r(2, 3);
    let r3 = r(1, 3);
    assert!(r1.contains(r2));
    assert!(!r1.contains(r3));
}

#[test]
fn intersection() {
    assert_eq!(TextRange::intersection(r(1, 2), r(2, 3)), Some(r(2, 2)));
    assert_eq!(TextRange::intersection(r(1, 5), r(2, 3)), Some(r(2, 3)));
    assert_eq!(TextRange::intersection(r(1, 2), r(3, 4)), None);
}

#[test]
fn covering() {
    assert_eq!(TextRange::covering(r(1, 2), r(2, 3)), r(1, 3));
    assert_eq!(TextRange::covering(r(1, 5), r(2, 3)), r(1, 5));
    assert_eq!(TextRange::covering(r(1, 2), r(4, 5)), r(1, 5));
}

#[test]
fn contains_point() {
    assert!(!r(1, 3).contains_point(0));
    assert!(r(1, 3).contains_point(1));
    assert!(r(1, 3).contains_point(2));
    assert!(!r(1, 3).contains_point(3));
    assert!(!r(1, 3).contains_point(4));

    assert!(!r(1, 3).contains_point_inclusive(0));
    assert!(r(1, 3).contains_point_inclusive(1));
    assert!(r(1, 3).contains_point_inclusive(2));
    assert!(r(1, 3).contains_point_inclusive(3));
    assert!(!r(1, 3).contains_point_inclusive(4));
}
