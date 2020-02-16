// For rust-lang/rust#68303: the contents of `UnsafeCell<T>` cannot
// participate in the niche-optimization for enum discriminants. This
// test checks that an `Option<UnsafeCell<NonZeroU32>>` has the same
// size in memory as an `Option<UnsafeCell<u32>>` (namely, 8 bytes).

// run-pass

#![feature(no_niche)]

use std::cell::UnsafeCell;
use std::mem::size_of;
use std::num::NonZeroU32 as N32;

struct Wrapper<T>(T);

#[repr(transparent)]
struct Transparent<T>(T);

#[repr(no_niche)]
struct NoNiche<T>(T);

fn main() {
    assert_eq!(size_of::<Option<Wrapper<u32>>>(),     8);
    assert_eq!(size_of::<Option<Wrapper<N32>>>(),     4);
    assert_eq!(size_of::<Option<Transparent<u32>>>(), 8);
    assert_eq!(size_of::<Option<Transparent<N32>>>(), 4);
    assert_eq!(size_of::<Option<NoNiche<u32>>>(),     8);
    assert_eq!(size_of::<Option<NoNiche<N32>>>(),     8);

    assert_eq!(size_of::<Option<UnsafeCell<u32>>>(),  8);
    assert_eq!(size_of::<Option<UnsafeCell<N32>>>(),  8);
}
