//@ check-pass

#![feature(new_range_api)]
#![feature(new_range)]

fn main() {
    // Unchanged
    let a: core::ops::RangeFull = ..;
    let b: core::ops::RangeTo<u8> = ..2;

    // FIXME: re-exports temporarily removed
    // let _: core::range::RangeFull = a;
    // let _: core::range::RangeTo<u8> = b;

    // Changed
    let a: core::range::RangeFrom<u8> = 1..;
    let b: core::range::Range<u8> = 2..3;
    let c: core::range::RangeInclusive<u8> = 4..=5;
    let d: core::range::RangeToInclusive<u8> = ..=3;

    let _: core::range::RangeFromIter<u8> = a.into_iter();
    let _: core::range::RangeIter<u8> = b.into_iter();
    let _: core::range::RangeInclusiveIter<u8> = c.into_iter();
    // RangeToInclusive has no Iterator implementation
}
