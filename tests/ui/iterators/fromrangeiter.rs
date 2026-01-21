//@ run-pass
//@ compile-flags: -C overflow-checks=yes

#![feature(new_range_api)]

use std::{iter, range};

fn main() {
    for (a, b) in iter::zip(0_u32..256, range::RangeFrom::from(0_u8..)) {
        assert_eq!(a, u32::from(b));
    }

    let mut a = range::RangeFrom::from(0_u8..).into_iter();
    let mut b = 0_u8..;
    assert_eq!(a.next(), b.next());
    assert_eq!(a.nth(5), b.nth(5));
    assert_eq!(a.nth(0), b.next());

    let mut a = range::RangeFrom::from(0_u8..).into_iter();
    let mut b = 0_u8..;
    assert_eq!(a.nth(5), b.nth(5));
    assert_eq!(a.nth(0), b.next());
}
