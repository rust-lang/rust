//@ run-pass
//@ compile-flags: -O -C debug_assertions=no

#![feature(new_range_api)]

fn main() {
    let mut it = core::range::RangeFrom::from(u8::MAX..).into_iter();
    assert_eq!(it.next().unwrap(), 255);
    assert_eq!(it.remainder().start, u8::MIN);
}
