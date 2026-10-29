//@ run-pass
//@ needs-unwind
//@ compile-flags: -O -C overflow-checks=yes

#![feature(new_range_remainder)]

fn main() {
    let mut it = core::range::RangeFrom::from(u8::MAX..).into_iter();
    assert_eq!(it.next().unwrap(), 255);
    assert!(it.next().is_none());
    assert!(it.remainder().is_none());

    let mut it = core::range::RangeFrom::from(u8::MAX..).into_iter();
    assert_eq!(it.next().unwrap(), 255);
    assert!(it.clone().remainder().is_none());
    assert!(it.next().is_none());

    let mut it = core::range::RangeFrom::from(u8::MAX..).into_iter();
    assert_eq!(it.next().unwrap(), 255);
    assert!(it.nth(0).is_none());

    let mut it = core::range::RangeFrom::from(u8::MAX-1..).into_iter();
    assert_eq!(it.nth(1).unwrap(), 255);
    assert!(it.next().is_none());

    let mut it = core::range::RangeFrom::from(u8::MAX-1..).into_iter();
    assert!(it.nth(2).is_none());
}
