//@ [full] known-bug: unknown

//@ revisions: full min
#![cfg_attr(full, feature(adt_const_params))]
#![cfg_attr(full, allow(incomplete_features))]

// `Range` should be usable within const generics:
struct _Range<const R: std::ops::Range<usize>>;
//[min]~^ ERROR `std::ops::Range<usize>` is forbidden
const RANGE: _Range<{ 0..1000 }> = _Range;

// `RangeFrom` should be usable within const generics:
struct _RangeFrom<const R: std::ops::RangeFrom<usize>>;
//[min]~^ ERROR `std::ops::RangeFrom<usize>` is forbidden
const RANGE_FROM: _RangeFrom<{ 0.. }> = _RangeFrom;

// `RangeFull` should be usable within const generics:
struct _RangeFull<const R: std::ops::RangeFull>;
//[min]~^ ERROR `RangeFull` is forbidden
const RANGE_FULL: _RangeFull<{ .. }> = _RangeFull;

// Regression test for #70155
// `RangeInclusive` should be usable within const generics:
struct _RangeInclusive<const R: std::ops::RangeInclusive<usize>>;
//[min]~^ ERROR `std::ops::RangeInclusive<usize>` is forbidden
const RANGE_INCLUSIVE: _RangeInclusive<{ 0..=999 }> = _RangeInclusive;

// `RangeTo` should be usable within const generics:
struct _RangeTo<const R: std::ops::RangeTo<usize>>;
//[min]~^ ERROR `RangeTo<usize>` is forbidden
const RANGE_TO: _RangeTo<{ ..1000 }> = _RangeTo;

// `RangeToInclusive` should be usable within const generics:
struct _RangeToInclusive<const R: std::ops::RangeToInclusive<usize>>;
//[min]~^ ERROR `std::ops::RangeToInclusive<usize>` is forbidden
const RANGE_TO_INCLUSIVE: _RangeToInclusive<{ ..=999 }> = _RangeToInclusive;

pub fn main() {}
