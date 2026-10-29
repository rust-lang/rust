//@ run-pass
//@ compile-flags: -O -C debug_assertions=no

#![feature(new_range_remainder)]

fn main() {
    let mut it = core::range::RangeFrom::from(u8::MAX..).into_iter();
    assert_eq!(it.next(), Some(255));
    assert_eq!(it.next(), None);
    assert!(it.remainder().is_none());
}
