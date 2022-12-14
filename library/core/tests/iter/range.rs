use super::*;

#[test]
fn test_range() {
    assert_eq!((0..5).collect::<Vec<_>>(), [0, 1, 2, 3, 4]);
    assert_eq!((-10..-1).collect::<Vec<_>>(), [-10, -9, -8, -7, -6, -5, -4, -3, -2]);
    assert_eq!((0..5).rev().collect::<Vec<_>>(), [4, 3, 2, 1, 0]);
    assert_eq!((200..-5).count(), 0);
    assert_eq!((200..-5).rev().count(), 0);
    assert_eq!((200..200).count(), 0);
    assert_eq!((200..200).rev().count(), 0);

    assert_eq!((0..100).size_hint(), (100, Some(100)));
    // this test is only meaningful when sizeof usize < sizeof u64
    assert_eq!((usize::MAX - 1..usize::MAX).size_hint(), (1, Some(1)));
    assert_eq!((-10..-1).size_hint(), (9, Some(9)));
    assert_eq!((-1..-10).size_hint(), (0, Some(0)));

    assert_eq!((-70..58).size_hint(), (128, Some(128)));
    assert_eq!((-128..127).size_hint(), (255, Some(255)));
    assert_eq!(
        (-2..isize::MAX).size_hint(),
        (isize::MAX as usize + 2, Some(isize::MAX as usize + 2))
    );
}

#[test]
fn test_char_range() {
    use std::char;
    // Miri is too slow
    let from = if cfg!(miri) { char::from_u32(0xD800 - 10).unwrap() } else { '\0' };
    let to = if cfg!(miri) { char::from_u32(0xDFFF + 10).unwrap() } else { char::MAX };
    assert!((from..=to).eq((from as u32..=to as u32).filter_map(char::from_u32)));
    assert!((from..=to).rev().eq((from as u32..=to as u32).filter_map(char::from_u32).rev()));

    assert_eq!(('\u{D7FF}'..='\u{E000}').count(), 2);
    assert_eq!(('\u{D7FF}'..='\u{E000}').size_hint(), (2, Some(2)));
    assert_eq!(('\u{D7FF}'..'\u{E000}').count(), 1);
    assert_eq!(('\u{D7FF}'..'\u{E000}').size_hint(), (1, Some(1)));
}

#[test]
fn test_range_exhaustion() {
    let mut r = 10..10;
    assert!(r.is_empty());
    assert_eq!(r.next(), None);
    assert_eq!(r.next_back(), None);
    assert_eq!(r, 10..10);

    let mut r = 10..12;
    assert_eq!(r.next(), Some(10));
    assert_eq!(r.next(), Some(11));
    assert!(r.is_empty());
    assert_eq!(r, 12..12);
    assert_eq!(r.next(), None);

    let mut r = 10..12;
    assert_eq!(r.next_back(), Some(11));
    assert_eq!(r.next_back(), Some(10));
    assert!(r.is_empty());
    assert_eq!(r, 10..10);
    assert_eq!(r.next_back(), None);

    let mut r = 100..10;
    assert!(r.is_empty());
    assert_eq!(r.next(), None);
    assert_eq!(r.next_back(), None);
    assert_eq!(r, 100..10);
}

#[test]
fn test_range_inclusive_exhaustion() {
    let mut r = 10..=10;
    assert_eq!(r.next(), Some(10));
    assert!(r.is_empty());
    assert_eq!(r.next(), None);
    assert_eq!(r.next(), None);

    assert_eq!(*r.start(), 10);
    assert_eq!(*r.end(), 10);
    assert_ne!(r, 10..=10);

    let mut r = 10..=10;
    assert_eq!(r.next_back(), Some(10));
    assert!(r.is_empty());
    assert_eq!(r.next_back(), None);

    assert_eq!(*r.start(), 10);
    assert_eq!(*r.end(), 10);
    assert_ne!(r, 10..=10);

    let mut r = 10..=12;
    assert_eq!(r.next(), Some(10));
    assert_eq!(r.next(), Some(11));
    assert_eq!(r.next(), Some(12));
    assert!(r.is_empty());
    assert_eq!(r.next(), None);

    let mut r = 10..=12;
    assert_eq!(r.next_back(), Some(12));
    assert_eq!(r.next_back(), Some(11));
    assert_eq!(r.next_back(), Some(10));
    assert!(r.is_empty());
    assert_eq!(r.next_back(), None);

    let mut r = 10..=12;
    assert_eq!(r.nth(2), Some(12));
    assert!(r.is_empty());
    assert_eq!(r.next(), None);

    let mut r = 10..=12;
    assert_eq!(r.nth(5), None);
    assert!(r.is_empty());
    assert_eq!(r.next(), None);

    let mut r = 100..=10;
    assert_eq!(r.next(), None);
    assert!(r.is_empty());
    assert_eq!(r.next(), None);
    assert_eq!(r.next(), None);
    assert_eq!(r, 100..=10);

    let mut r = 100..=10;
    assert_eq!(r.next_back(), None);
    assert!(r.is_empty());
    assert_eq!(r.next_back(), None);
    assert_eq!(r.next_back(), None);
    assert_eq!(r, 100..=10);
}

#[test]
fn test_range_nth() {
    assert_eq!((10..15).nth(0), Some(10));
    assert_eq!((10..15).nth(1), Some(11));
    assert_eq!((10..15).nth(4), Some(14));
    assert_eq!((10..15).nth(5), None);

    let mut r = 10..20;
    assert_eq!(r.nth(2), Some(12));
    assert_eq!(r, 13..20);
    assert_eq!(r.nth(2), Some(15));
    assert_eq!(r, 16..20);
    assert_eq!(r.nth(10), None);
    assert_eq!(r, 20..20);
}

#[test]
fn test_range_nth_back() {
    assert_eq!((10..15).nth_back(0), Some(14));
    assert_eq!((10..15).nth_back(1), Some(13));
    assert_eq!((10..15).nth_back(4), Some(10));
    assert_eq!((10..15).nth_back(5), None);
    assert_eq!((-120..80_i8).nth_back(199), Some(-120));

    let mut r = 10..20;
    assert_eq!(r.nth_back(2), Some(17));
    assert_eq!(r, 10..17);
    assert_eq!(r.nth_back(2), Some(14));
    assert_eq!(r, 10..14);
    assert_eq!(r.nth_back(10), None);
    assert_eq!(r, 10..10);
}

#[test]
fn test_range_from_nth() {
    assert_eq!((10..).nth(0), Some(10));
    assert_eq!((10..).nth(1), Some(11));
    assert_eq!((10..).nth(4), Some(14));

    let mut r = 10..;
    assert_eq!(r.nth(2), Some(12));
    assert_eq!(r, 13..);
    assert_eq!(r.nth(2), Some(15));
    assert_eq!(r, 16..);
    assert_eq!(r.nth(10), Some(26));
    assert_eq!(r, 27..);

    assert_eq!((0..).size_hint(), (usize::MAX, None));
}

#[test]
fn test_range_from_take() {
    let mut it = (0..).take(3);
    assert_eq!(it.next(), Some(0));
    assert_eq!(it.next(), Some(1));
    assert_eq!(it.next(), Some(2));
    assert_eq!(it.next(), None);
    is_trusted_len((0..).take(3));
    assert_eq!((0..).take(3).size_hint(), (3, Some(3)));
    assert_eq!((0..).take(0).size_hint(), (0, Some(0)));
    assert_eq!((0..).take(usize::MAX).size_hint(), (usize::MAX, Some(usize::MAX)));
}

#[test]
fn test_range_from_take_collect() {
    let v: Vec<_> = (0..).take(3).collect();
    assert_eq!(v, vec![0, 1, 2]);
}

#[test]
fn test_range_inclusive_nth() {
    assert_eq!((10..=15).nth(0), Some(10));
    assert_eq!((10..=15).nth(1), Some(11));
    assert_eq!((10..=15).nth(5), Some(15));
    assert_eq!((10..=15).nth(6), None);

    let mut exhausted_via_next = 10_u8..=20;
    while exhausted_via_next.next().is_some() {}

    let mut r = 10_u8..=20;
    assert_eq!(r.nth(2), Some(12));
    assert_eq!(r, 13..=20);
    assert_eq!(r.nth(2), Some(15));
    assert_eq!(r, 16..=20);
    assert_eq!(r.is_empty(), false);
    assert_eq!(ExactSizeIterator::is_empty(&r), false);
    assert_eq!(r.nth(10), None);
    assert_eq!(r.is_empty(), true);
    assert_eq!(r, exhausted_via_next);
    assert_eq!(ExactSizeIterator::is_empty(&r), true);
}

#[test]
fn test_range_inclusive_nth_back() {
    assert_eq!((10..=15).nth_back(0), Some(15));
    assert_eq!((10..=15).nth_back(1), Some(14));
    assert_eq!((10..=15).nth_back(5), Some(10));
    assert_eq!((10..=15).nth_back(6), None);
    assert_eq!((-120..=80_i8).nth_back(200), Some(-120));

    let mut exhausted_via_next_back = 10_u8..=20;
    while exhausted_via_next_back.next_back().is_some() {}

    let mut r = 10_u8..=20;
    assert_eq!(r.nth_back(2), Some(18));
    assert_eq!(r, 10..=17);
    assert_eq!(r.nth_back(2), Some(15));
    assert_eq!(r, 10..=14);
    assert_eq!(r.is_empty(), false);
    assert_eq!(ExactSizeIterator::is_empty(&r), false);
    assert_eq!(r.nth_back(10), None);
    assert_eq!(r.is_empty(), true);
    assert_eq!(r, exhausted_via_next_back);
    assert_eq!(ExactSizeIterator::is_empty(&r), true);
}

#[test]
fn test_range_len() {
    assert_eq!((0..10_u8).len(), 10);
    assert_eq!((9..10_u8).len(), 1);
    assert_eq!((10..10_u8).len(), 0);
    assert_eq!((11..10_u8).len(), 0);
    assert_eq!((100..10_u8).len(), 0);
}

#[test]
fn test_range_inclusive_len() {
    assert_eq!((0..=10_u8).len(), 11);
    assert_eq!((9..=10_u8).len(), 2);
    assert_eq!((10..=10_u8).len(), 1);
    assert_eq!((11..=10_u8).len(), 0);
    assert_eq!((100..=10_u8).len(), 0);
}

#[test]
fn test_range_step() {
    #![allow(deprecated)]

    assert_eq!((0..20).step_by(5).collect::<Vec<isize>>(), [0, 5, 10, 15]);
    assert_eq!((1..21).rev().step_by(5).collect::<Vec<isize>>(), [20, 15, 10, 5]);
    assert_eq!((1..21).rev().step_by(6).collect::<Vec<isize>>(), [20, 14, 8, 2]);
    assert_eq!((200..255).step_by(50).collect::<Vec<u8>>(), [200, 250]);
    assert_eq!((200..-5).step_by(1).collect::<Vec<isize>>(), []);
    assert_eq!((200..200).step_by(1).collect::<Vec<isize>>(), []);

    assert_eq!((0..20).step_by(1).size_hint(), (20, Some(20)));
    assert_eq!((0..20).step_by(21).size_hint(), (1, Some(1)));
    assert_eq!((0..20).step_by(5).size_hint(), (4, Some(4)));
    assert_eq!((1..21).rev().step_by(5).size_hint(), (4, Some(4)));
    assert_eq!((1..21).rev().step_by(6).size_hint(), (4, Some(4)));
    assert_eq!((20..-5).step_by(1).size_hint(), (0, Some(0)));
    assert_eq!((20..20).step_by(1).size_hint(), (0, Some(0)));
    assert_eq!((i8::MIN..i8::MAX).step_by(-(i8::MIN as i32) as usize).size_hint(), (2, Some(2)));
    assert_eq!((i16::MIN..i16::MAX).step_by(i16::MAX as usize).size_hint(), (3, Some(3)));
    assert_eq!((isize::MIN..isize::MAX).step_by(1).size_hint(), (usize::MAX, Some(usize::MAX)));
}

#[test]
fn test_range_advance_by() {
    let mut r = 0..usize::MAX;
    r.advance_by(0).unwrap();
    r.advance_back_by(0).unwrap();

    assert_eq!(r.len(), usize::MAX);

    r.advance_by(1).unwrap();
    r.advance_back_by(1).unwrap();

    assert_eq!((r.start, r.end), (1, usize::MAX - 1));

    assert_eq!(r.advance_by(usize::MAX), Err(usize::MAX - 2));

    r.advance_by(0).unwrap();
    r.advance_back_by(0).unwrap();

    let mut r = 0u128..u128::MAX;

    r.advance_by(usize::MAX).unwrap();
    r.advance_back_by(usize::MAX).unwrap();

    assert_eq!((r.start, r.end), (0u128 + usize::MAX as u128, u128::MAX - usize::MAX as u128));
}

#[test]
fn test_range_inclusive_step() {
    assert_eq!((0..=50).step_by(10).collect::<Vec<_>>(), [0, 10, 20, 30, 40, 50]);
    assert_eq!((0..=5).step_by(1).collect::<Vec<_>>(), [0, 1, 2, 3, 4, 5]);
    assert_eq!((200..=255u8).step_by(10).collect::<Vec<_>>(), [200, 210, 220, 230, 240, 250]);
    assert_eq!((250..=255u8).step_by(1).collect::<Vec<_>>(), [250, 251, 252, 253, 254, 255]);
}

#[test]
fn test_range_last_max() {
    assert_eq!((0..20).last(), Some(19));
    assert_eq!((-20..0).last(), Some(-1));
    assert_eq!((5..5).last(), None);

    assert_eq!((0..20).max(), Some(19));
    assert_eq!((-20..0).max(), Some(-1));
    assert_eq!((5..5).max(), None);
}

#[test]
fn test_range_inclusive_last_max() {
    assert_eq!((0..=20).last(), Some(20));
    assert_eq!((-20..=0).last(), Some(0));
    assert_eq!((5..=5).last(), Some(5));
    let mut r = 10..=10;
    r.next();
    assert_eq!(r.last(), None);

    assert_eq!((0..=20).max(), Some(20));
    assert_eq!((-20..=0).max(), Some(0));
    assert_eq!((5..=5).max(), Some(5));
    let mut r = 10..=10;
    r.next();
    assert_eq!(r.max(), None);
}

#[test]
fn test_range_min() {
    assert_eq!((0..20).min(), Some(0));
    assert_eq!((-20..0).min(), Some(-20));
    assert_eq!((5..5).min(), None);
}

#[test]
fn test_range_inclusive_min() {
    assert_eq!((0..=20).min(), Some(0));
    assert_eq!((-20..=0).min(), Some(-20));
    assert_eq!((5..=5).min(), Some(5));
    let mut r = 10..=10;
    r.next();
    assert_eq!(r.min(), None);
}

#[test]
fn test_range_inclusive_folds() {
    assert_eq!((1..=10).sum::<i32>(), 55);
    assert_eq!((1..=10).rev().sum::<i32>(), 55);

    let mut it = 44..=50;
    assert_eq!(it.try_fold(0, i8::checked_add), None);
    assert_eq!(it, 47..=50);
    assert_eq!(it.try_fold(0, i8::checked_add), None);
    assert_eq!(it, 50..=50);
    assert_eq!(it.try_fold(0, i8::checked_add), Some(50));
    assert!(it.is_empty());
    assert_eq!(it.try_fold(0, i8::checked_add), Some(0));
    assert!(it.is_empty());

    let mut it = 40..=47;
    assert_eq!(it.try_rfold(0, i8::checked_add), None);
    assert_eq!(it, 40..=44);
    assert_eq!(it.try_rfold(0, i8::checked_add), None);
    assert_eq!(it, 40..=41);
    assert_eq!(it.try_rfold(0, i8::checked_add), Some(81));
    assert!(it.is_empty());
    assert_eq!(it.try_rfold(0, i8::checked_add), Some(0));
    assert!(it.is_empty());

    let mut it = 10..=20;
    assert_eq!(it.try_fold(0, |a, b| Some(a + b)), Some(165));
    assert!(it.is_empty());
    assert_eq!(it.try_fold(0, |a, b| Some(a + b)), Some(0));
    assert!(it.is_empty());

    let mut it = 10..=20;
    assert_eq!(it.try_rfold(0, |a, b| Some(a + b)), Some(165));
    assert!(it.is_empty());
    assert_eq!(it.try_rfold(0, |a, b| Some(a + b)), Some(0));
    assert!(it.is_empty());
}

#[test]
fn test_range_size_hint() {
    assert_eq!((0..0usize).size_hint(), (0, Some(0)));
    assert_eq!((0..100usize).size_hint(), (100, Some(100)));
    assert_eq!((0..usize::MAX).size_hint(), (usize::MAX, Some(usize::MAX)));

    let umax = u128::try_from(usize::MAX).unwrap();
    assert_eq!((0..0u128).size_hint(), (0, Some(0)));
    assert_eq!((0..100u128).size_hint(), (100, Some(100)));
    assert_eq!((0..umax).size_hint(), (usize::MAX, Some(usize::MAX)));
    assert_eq!((0..umax + 1).size_hint(), (usize::MAX, None));

    assert_eq!((0..0isize).size_hint(), (0, Some(0)));
    assert_eq!((-100..100isize).size_hint(), (200, Some(200)));
    assert_eq!((isize::MIN..isize::MAX).size_hint(), (usize::MAX, Some(usize::MAX)));

    let imin = i128::try_from(isize::MIN).unwrap();
    let imax = i128::try_from(isize::MAX).unwrap();
    assert_eq!((0..0i128).size_hint(), (0, Some(0)));
    assert_eq!((-100..100i128).size_hint(), (200, Some(200)));
    assert_eq!((imin..imax).size_hint(), (usize::MAX, Some(usize::MAX)));
    assert_eq!((imin..imax + 1).size_hint(), (usize::MAX, None));
}

#[test]
fn test_range_inclusive_size_hint() {
    assert_eq!((1..=0usize).size_hint(), (0, Some(0)));
    assert_eq!((0..=0usize).size_hint(), (1, Some(1)));
    assert_eq!((0..=100usize).size_hint(), (101, Some(101)));
    assert_eq!((0..=usize::MAX - 1).size_hint(), (usize::MAX, Some(usize::MAX)));
    assert_eq!((0..=usize::MAX).size_hint(), (usize::MAX, None));

    let umax = u128::try_from(usize::MAX).unwrap();
    assert_eq!((1..=0u128).size_hint(), (0, Some(0)));
    assert_eq!((0..=0u128).size_hint(), (1, Some(1)));
    assert_eq!((0..=100u128).size_hint(), (101, Some(101)));
    assert_eq!((0..=umax - 1).size_hint(), (usize::MAX, Some(usize::MAX)));
    assert_eq!((0..=umax).size_hint(), (usize::MAX, None));
    assert_eq!((0..=umax + 1).size_hint(), (usize::MAX, None));

    assert_eq!((0..=-1isize).size_hint(), (0, Some(0)));
    assert_eq!((0..=0isize).size_hint(), (1, Some(1)));
    assert_eq!((-100..=100isize).size_hint(), (201, Some(201)));
    assert_eq!((isize::MIN..=isize::MAX - 1).size_hint(), (usize::MAX, Some(usize::MAX)));
    assert_eq!((isize::MIN..=isize::MAX).size_hint(), (usize::MAX, None));

    let imin = i128::try_from(isize::MIN).unwrap();
    let imax = i128::try_from(isize::MAX).unwrap();
    assert_eq!((0..=-1i128).size_hint(), (0, Some(0)));
    assert_eq!((0..=0i128).size_hint(), (1, Some(1)));
    assert_eq!((-100..=100i128).size_hint(), (201, Some(201)));
    assert_eq!((imin..=imax - 1).size_hint(), (usize::MAX, Some(usize::MAX)));
    assert_eq!((imin..=imax).size_hint(), (usize::MAX, None));
    assert_eq!((imin..=imax + 1).size_hint(), (usize::MAX, None));
}

#[test]
fn test_double_ended_range() {
    assert_eq!((11..14).rev().collect::<Vec<_>>(), [13, 12, 11]);
    for _ in (10..0).rev() {
        panic!("unreachable");
    }

    assert_eq!((11..14).rev().collect::<Vec<_>>(), [13, 12, 11]);
    for _ in (10..0).rev() {
        panic!("unreachable");
    }
}
