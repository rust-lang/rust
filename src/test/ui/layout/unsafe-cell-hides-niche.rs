// For rust-lang/rust#68303: the contents of `UnsafeCell<T>` cannot
// participate in the niche-optimization for enum discriminants. This
// test checks that an `Option<UnsafeCell<NonZeroU32>>` has the same
// size in memory as an `Option<UnsafeCell<u32>>` (namely, 8 bytes).

// run-pass

#![feature(repr_simd)]

use std::cell::{UnsafeCell, RefCell, Cell};
use std::mem::size_of;
use std::num::NonZeroU32 as N32;
use std::sync::{Mutex, RwLock};

struct Wrapper<T>(T);

#[repr(transparent)]
struct Transparent<T>(T);

struct NoNiche<T>(UnsafeCell<T>);

fn main() {
    assert_eq!(size_of::<Option<Wrapper<u32>>>(),     8);
    assert_eq!(size_of::<Option<Wrapper<N32>>>(),     4); // (✓ niche opt)
    assert_eq!(size_of::<Option<Transparent<u32>>>(), 8);
    assert_eq!(size_of::<Option<Transparent<N32>>>(), 4); // (✓ niche opt)
    assert_eq!(size_of::<Option<NoNiche<u32>>>(),     8);
    assert_eq!(size_of::<Option<NoNiche<N32>>>(),     8); // (✗ niche opt)

    assert_eq!(size_of::<Option<UnsafeCell<u32>>>(),  8);
    assert_eq!(size_of::<Option<UnsafeCell<N32>>>(),  8); // (✗ niche opt)

    assert_eq!(size_of::<       UnsafeCell<&()> >(), 8);
    assert_eq!(size_of::<Option<UnsafeCell<&()>>>(), 16); // (✗ niche opt)
    assert_eq!(size_of::<             Cell<&()> >(), 8);
    assert_eq!(size_of::<Option<      Cell<&()>>>(), 16); // (✗ niche opt)
    assert_eq!(size_of::<          RefCell<&()> >(), 16);
    assert_eq!(size_of::<Option<   RefCell<&()>>>(), 24); // (✗ niche opt)
    assert_eq!(size_of::<           RwLock<&()> >(), 24);
    assert_eq!(size_of::<Option<    RwLock<&()>>>(), 32); // (✗ niche opt)
    assert_eq!(size_of::<            Mutex<&()> >(), 16);
    assert_eq!(size_of::<Option<     Mutex<&()>>>(), 24); // (✗ niche opt)

    assert_eq!(size_of::<       UnsafeCell<&[i32]> >(), 16);
    assert_eq!(size_of::<Option<UnsafeCell<&[i32]>>>(), 24); // (✗ niche opt)
    assert_eq!(size_of::<       UnsafeCell<(&(), &())> >(), 16);
    assert_eq!(size_of::<Option<UnsafeCell<(&(), &())>>>(), 24); // (✗ niche opt)

    trait Trait {}
    assert_eq!(size_of::<       UnsafeCell<&dyn Trait> >(), 16);
    assert_eq!(size_of::<Option<UnsafeCell<&dyn Trait>>>(), 24); // (✗ niche opt)

    #[repr(simd)]
    pub struct Vec4<T>([T; 4]);

    assert_eq!(size_of::<       UnsafeCell<Vec4<N32>> >(), 16);
    assert_eq!(size_of::<Option<UnsafeCell<Vec4<N32>>>>(), 32); // (✗ niche opt)
}
