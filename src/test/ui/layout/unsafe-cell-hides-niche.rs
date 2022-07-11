// For rust-lang/rust#68303: the contents of `UnsafeCell<T>` cannot
// participate in the niche-optimization for enum discriminants. This
// test checks that an `Option<UnsafeCell<NonZeroU32>>` has the same
// size in memory as an `Option<UnsafeCell<u32>>` (namely, 8 bytes).

// check-pass

#![feature(repr_simd)]

use std::cell::{UnsafeCell, RefCell, Cell};
use std::num::NonZeroU32 as N32;
use std::sync::{Mutex, RwLock};

struct Wrapper<T>(T);

#[repr(transparent)]
struct Transparent<T>(T);

struct NoNiche<T>(UnsafeCell<T>);

// Overwriting the runtime assertion and making it a compile-time assertion
macro_rules! assert_eq {
    ($a:ty, $b:literal) => {{
        const _: () = assert!(std::mem::size_of::<$a>() == $b);
    }};
}

fn main() {
    assert_eq!(Option<Wrapper<u32>>,     8);
    assert_eq!(Option<Wrapper<N32>>,     4); // (✓ niche opt)
    assert_eq!(Option<Transparent<u32>>, 8);
    assert_eq!(Option<Transparent<N32>>, 4); // (✓ niche opt)
    assert_eq!(Option<NoNiche<u32>>,     8);
    assert_eq!(Option<NoNiche<N32>>,     8); // (✗ niche opt)

    assert_eq!(Option<UnsafeCell<u32>>,  8);
    assert_eq!(Option<UnsafeCell<N32>>,  8); // (✗ niche opt)

    assert_eq!(       UnsafeCell<&()> , 8);
    assert_eq!(Option<UnsafeCell<&()>>, 16); // (✗ niche opt)
    assert_eq!(             Cell<&()> , 8);
    assert_eq!(Option<      Cell<&()>>, 16); // (✗ niche opt)
    assert_eq!(          RefCell<&()> , 16);
    assert_eq!(Option<   RefCell<&()>>, 24); // (✗ niche opt)
    assert_eq!(           RwLock<&()> , 24);
    assert_eq!(Option<    RwLock<&()>>, 32); // (✗ niche opt)
    assert_eq!(            Mutex<&()> , 16);
    assert_eq!(Option<     Mutex<&()>>, 24); // (✗ niche opt)

    assert_eq!(       UnsafeCell<&[i32]> , 16);
    assert_eq!(Option<UnsafeCell<&[i32]>>, 24); // (✗ niche opt)
    assert_eq!(       UnsafeCell<(&(), &())> , 16);
    assert_eq!(Option<UnsafeCell<(&(), &())>>, 24); // (✗ niche opt)

    trait Trait {}
    assert_eq!(       UnsafeCell<&dyn Trait> , 16);
    assert_eq!(Option<UnsafeCell<&dyn Trait>>, 24); // (✗ niche opt)

    #[repr(simd)]
    pub struct Vec4<T>([T; 4]);

    assert_eq!(       UnsafeCell<Vec4<N32>> , 16);
    assert_eq!(Option<UnsafeCell<Vec4<N32>>>, 32); // (✗ niche opt)
}
