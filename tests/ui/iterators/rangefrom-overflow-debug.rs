//@ run-pass
//@ needs-unwind
//@ compile-flags: -O -C debug_assertions=yes

#![feature(new_range_api)]

use std::panic;

fn main() {
    let mut it = core::range::RangeFrom::from(u8::MAX..).into_iter();
    assert_eq!(it.next().unwrap(), 255);

    let r = panic::catch_unwind(|| {
        let _ = it.remainder();
    });
    assert!(r.is_err());
}
