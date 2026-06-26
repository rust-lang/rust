//@ run-pass
//@ needs-unwind
//@ aux-build:rangefrom-overflow-2crates-ocno.rs
//@ aux-build:rangefrom-overflow-2crates-ocyes.rs

// For #154124
// Test that two crates with different overflow-checks have the same results,
// even when the iterator is passed between them.

extern crate rangefrom_overflow_2crates_ocno;
extern crate rangefrom_overflow_2crates_ocyes;

use rangefrom_overflow_2crates_ocno::next as next_ocno;
use rangefrom_overflow_2crates_ocyes::next as next_ocyes;

fn main() {
    let mut iter_ocyes = std::range::RangeFrom::from(0_u8..).into_iter();
    let mut iter_ocno = iter_ocyes.clone();

    for n in 0_u8..=255 {
        assert_eq!(n, next_ocno(&mut iter_ocyes.clone()));
        assert_eq!(n, next_ocyes(&mut iter_ocyes));
        assert_eq!(n, next_ocyes(&mut iter_ocno.clone()));
        assert_eq!(n, next_ocno(&mut iter_ocno));
    }

    // `iter_ocno` should have wrapped
    assert_eq!(0, next_ocyes(&mut iter_ocno.clone()));
    assert_eq!(0, next_ocno(&mut iter_ocno));
    // `iter_ocyes` should be exhausted,
    // which will wrap when called without overflow-checks
    assert_eq!(0, next_ocno(&mut iter_ocyes.clone()));
    // and panic when called with overflow-checks
    let r = std::panic::catch_unwind(move || {
        let _ = next_ocyes(&mut iter_ocyes);
    });
    assert!(r.is_err());
}
