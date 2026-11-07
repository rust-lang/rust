// For rust-lang/rust#68303: the contents of `UnsafeCell<T>` cannot
// participate in the niche-optimization for enum discriminants. This
// test checks that an `Option<UnsafeCell<NonZero<u32>>>` has the same
// size in memory as an `Option<UnsafeCell<u32>>` (namely, 8 bytes).
//
//@ check-pass
//@ compile-flags: --crate-type=lib
//@ only-x86
#![feature(repr_simd)]

use std::cell::{UnsafeCell, RefCell, Cell};
use std::mem::size_of;
use std::num::NonZero;
use std::sync::{Mutex, RwLock};

struct Wrapper<T>(#[allow(dead_code)] T);

#[repr(transparent)]
struct Transparent<T>(T);

struct NoNiche<T>(UnsafeCell<T>);

struct Size<const S: usize>;

macro_rules! check_sizes {
    (check_one_specific_size: $ty:ty, $size:expr) => {
        const _: Size::<{$size}> = Size::<{size_of::<$ty>()}>;
    };
    // Any tests run on `UnsafeCell` must be the same for `Cell`
    (UnsafeCell<$ty:ty>: $size:expr => $optioned_size:expr) => {
        check_sizes!(Cell<$ty>: $size => $optioned_size);
        check_sizes!(@actual_check: UnsafeCell<$ty>: $size => $optioned_size);
    };
    ($ty:ty: $size:expr => $optioned_size:expr) => {
        check_sizes!(@actual_check: $ty: $size => $optioned_size);
    };
    // This branch does the actual checking logic, the `@actual_check` prefix is here to distinguish
    // it from other branches and not accidentally match any.
    (@actual_check: $ty:ty: $size:expr => $optioned_size:expr) => {
        check_sizes!(check_one_specific_size: $ty, $size);
        check_sizes!(check_one_specific_size: Option<$ty>, $optioned_size);
        check_sizes!(check_no_niche_opt: $size != $optioned_size, $ty);
    };
    // only check that there is no niche (size goes up when wrapped in an option),
    // don't check actual sizes
    ($ty:ty) => {
        check_sizes!(check_no_niche_opt: true, $ty);
    };
    (check_no_niche_opt: $no_niche_opt:expr, $ty:ty) => {
        const _: () = if $no_niche_opt { assert!(size_of::<$ty>() < size_of::<Option<$ty>>()); };
    };
}

const PTR_SIZE: usize = std::mem::size_of::<*const ()>();

check_sizes!(Wrapper<u32>:              4 => 8);
check_sizes!(Wrapper<NonZero<u32>>:     4 => 4); // (✓ niche opt)

check_sizes!(Transparent<u32>:          4 => 8);
check_sizes!(Transparent<NonZero<u32>>: 4 => 4); // (✓ niche opt)

check_sizes!(NoNiche<u32>:              4 => 8);
check_sizes!(NoNiche<NonZero<u32>>:     4 => 8);

check_sizes!(UnsafeCell<u32>:           4 => 8);
check_sizes!(UnsafeCell<NonZero<u32>>:  4 => 8);

check_sizes!(UnsafeCell<&()>: PTR_SIZE => PTR_SIZE * 2);
check_sizes!(   RefCell<&()>: PTR_SIZE * 2 => PTR_SIZE * 3);

check_sizes!(RwLock<&()>);
check_sizes!(Mutex<&()>);

check_sizes!(UnsafeCell<&[i32]>: PTR_SIZE * 2 => PTR_SIZE * 3);
check_sizes!(UnsafeCell<(&(), &())>: PTR_SIZE * 2 => PTR_SIZE * 3);

trait Trait {}
check_sizes!(UnsafeCell<&dyn Trait>: PTR_SIZE * 2 => PTR_SIZE * 3);

#[repr(simd)]
pub struct Vec4<T>([T; 4]);

check_sizes!(UnsafeCell<Vec4<NonZero<u32>>>: 16 => 32);
