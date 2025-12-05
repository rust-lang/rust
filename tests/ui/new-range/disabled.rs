//@ check-pass

#![feature(new_range_api)]

fn main() {
    // Unchanged
    let a: core::range::RangeFull = ..;
    let b: core::range::RangeTo<u8> = ..2;

    let _: core::ops::RangeFull = a;
    let _: core::ops::RangeTo<u8> = b;

    // Changed
    let a: core::range::legacy::RangeFrom<u8> = 1..;
    let b: core::range::legacy::Range<u8> = 2..3;
    let c: core::range::legacy::RangeInclusive<u8> = 4..=5;
    let d: core::range::legacy::RangeToInclusive<u8> = ..=3;

    let a: core::ops::RangeFrom<u8> = a;
    let b: core::ops::Range<u8> = b;
    let c: core::ops::RangeInclusive<u8> = c;
    let d: core::ops::RangeToInclusive<u8> = d;

    let _: core::ops::RangeFrom<u8> = a.into_iter();
    let _: core::ops::Range<u8> = b.into_iter();
    let _: core::ops::RangeInclusive<u8> = c.into_iter();
}
