// For rust-lang/rust#68303: the contents of `UnsafeCell<T>` cannot
// participate in the niche-optimization for enum discriminants. This
// test checks that an `Option<UnsafeCell<NonZeroU32>>` has the same
// size in memory as an `Option<UnsafeCell<u32>>` (namely, 8 bytes).

// check-pass
// compile-flags: --crate-type=lib

#![feature(repr_simd)]

use std::cell::{UnsafeCell, RefCell, Cell};
use std::mem::size_of;
use std::num::NonZeroU32 as N32;
use std::sync::{Mutex, RwLock};

struct Wrapper<T>(T);

#[repr(transparent)]
struct Transparent<T>(T);

struct NoNiche<T>(UnsafeCell<T>);

struct Size<const S: usize>;

// Overwriting the runtime assertion and making it a compile-time assertion
macro_rules! assert_size_eq {
    ($ty:ty, $size:expr) => {
        const _: Size::<{$size}> = Size::<{size_of::<$ty>()}>;
    };
    ($ty:ty, $size:expr, $optioned_size:expr) => {
        assert_size_eq!($ty, $size);
        assert_size_eq!(Option<$ty>, $optioned_size);
        const _: () = assert!(
            $size == $optioned_size ||
            size_of::<$ty>() < size_of::<Option<$ty>>()
        );
    };
}

const PTR_SIZE: usize = std::mem::size_of::<*const ()>();

assert_size_eq!(Wrapper<u32>,     4, 8);
assert_size_eq!(Wrapper<N32>,     4, 4); // (✓ niche opt)
assert_size_eq!(Transparent<u32>, 4, 8);
assert_size_eq!(Transparent<N32>, 4, 4); // (✓ niche opt)
assert_size_eq!(NoNiche<u32>,     4, 8);
assert_size_eq!(NoNiche<N32>,     4, 8);

assert_size_eq!(UnsafeCell<u32>,  4, 8);
assert_size_eq!(UnsafeCell<N32>,  4, 8);

assert_size_eq!(UnsafeCell<&()> , PTR_SIZE, PTR_SIZE * 2);
assert_size_eq!(      Cell<&()> , PTR_SIZE, PTR_SIZE * 2);
assert_size_eq!(   RefCell<&()> , PTR_SIZE * 2, PTR_SIZE * 3);
assert_size_eq!(
    RwLock<&()>,
    if cfg!(target_pointer_width = "32") { 16 } else { 24 },
    if cfg!(target_pointer_width = "32") { 20 } else { 32 }
);
assert_size_eq!(
    Mutex<&()> ,
    if cfg!(target_pointer_width = "32") { 12 } else { 16 },
    if cfg!(target_pointer_width = "32") { 16 } else { 24 }
);

assert_size_eq!(UnsafeCell<&[i32]> , PTR_SIZE * 2, PTR_SIZE * 3);
assert_size_eq!(UnsafeCell<(&(), &())> , PTR_SIZE * 2, PTR_SIZE * 3);

trait Trait {}
assert_size_eq!(UnsafeCell<&dyn Trait> , PTR_SIZE * 2, PTR_SIZE * 3);

#[repr(simd)]
pub struct Vec4<T>([T; 4]);

assert_size_eq!(UnsafeCell<Vec4<N32>> , 16, 32);
