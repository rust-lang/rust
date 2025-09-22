//@ check-pass

#![feature(new_range_api)]
#![feature(new_range)]

fn main() {
    // Unchanged
    let a: core::range::RangeFull = ..;
    let b: core::range::RangeTo<u8> = ..2;

    let _: core::ops::RangeFull = a;
    let _: core::ops::RangeTo<u8> = b;

    // Changed
    let a: core::range::RangeFrom<u8> = 1..;
    let b: core::range::Range<u8> = 2..3;
    let c: core::range::RangeInclusive<u8> = 4..=5;
    let d: core::range::RangeToInclusive<u8> = ..=3;

    let _: core::range::IterRangeFrom<u8> = a.into_iter();
    let _: core::range::IterRange<u8> = b.into_iter();
    let _: core::range::IterRangeInclusive<u8> = c.into_iter();
    // RangeToInclusive has no Iterator implementation
}
