//@ run-pass
//@ needs-unwind
//@ compile-flags: -O -C overflow-checks=yes

#![feature(new_range_api)]

use std::panic;

fn main() {
    let mut it = core::range::RangeFrom::from(u8::MAX..).into_iter();
    assert_eq!(it.next().unwrap(), 255);

    let r = panic::catch_unwind(move || {
        let _ = it.remainder();
    });
    assert!(r.is_err());

    let mut it = core::range::RangeFrom::from(u8::MAX..).into_iter();
    assert_eq!(it.next().unwrap(), 255);

    let r = panic::catch_unwind(move || {
        let _ = it.next();
    });
    assert!(r.is_err());

    let mut it = core::range::RangeFrom::from(u8::MAX..).into_iter();
    assert_eq!(it.next().unwrap(), 255);

    let r = panic::catch_unwind(move || {
        let _ = it.nth(0);
    });
    assert!(r.is_err());

    let mut it = core::range::RangeFrom::from(u8::MAX-1..).into_iter();
    assert_eq!(it.nth(1).unwrap(), 255);

    let r = panic::catch_unwind(move || {
        let _ = it.next();
    });
    assert!(r.is_err());

    let mut it = core::range::RangeFrom::from(u8::MAX-1..).into_iter();

    let r = panic::catch_unwind(move || {
        let _ = it.nth(2);
    });
    assert!(r.is_err());
}
