#![feature(rustc_attrs)]
#![feature(never_type)]
#![crate_type = "lib"]

// Regression test for https://github.com/rust-lang/rust/issues/159438
// where with `i8` it was getting an unnecessarily-broad `valid_range`

#[rustc_dump_layout(largest_niche)]
enum With128Variants {
    //~^ ERROR: value: u8, valid_range: 0..=127
    _0 = 0,
    _1 = 1,
    _2 = 2,
    // layout doesn't actually care if we elide a bunch here
    _125 = 125,
    _126 = 126,
    _127 = 127,
}

#[rustc_dump_layout(largest_niche)]
#[repr(i8)]
enum With128VariantsI8 {
    //~^ ERROR: value: i8, valid_range: (..=2) | (125..)
    _0 = 0,
    _1 = 1,
    _2 = 2,
    // layout doesn't actually care if we elide a bunch here
    _125 = 125,
    _126 = 126,
    _127 = 127,
}

#[rustc_dump_layout(largest_niche)]
#[repr(u8)]
enum With128VariantsU8 {
    //~^ ERROR: value: u8, valid_range: 0..=127
    _0 = 0,
    _1 = 1,
    _2 = 2,
    // layout doesn't actually care if we elide a bunch here
    _125 = 125,
    _126 = 126,
    _127 = 127,
}

// For these either the wrapping or the non-wrapping `valid_range` have the same size,
// but it would be nice to consistently pick the one that leaves `0` unclaimed
// so that it can be used for `None` later.

#[rustc_dump_layout(largest_niche)]
enum Symmetric {
    //~^ ERROR: value: u8, valid_range: 1..=129
    A = 1,
    B = 129,
}

#[rustc_dump_layout(largest_niche)]
#[repr(u8)]
enum SymmetricU8 {
    //~^ ERROR: value: u8, valid_range: (..=1) | (129..)
    A = 1,
    B = 129,
}

#[rustc_dump_layout(largest_niche)]
enum SymmetricSigned {
    //~^ ERROR: value: i8, valid_range: (..=1) | (129..)
    A = -127,
    B = 1,
}

#[rustc_dump_layout(largest_niche)]
#[repr(i8)]
enum SymmetricSignedI8 {
    //~^ ERROR: value: i8, valid_range: (..=1) | (129..)
    A = -127,
    B = 1,
}
