// Copyright 2012-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Basic functions for dealing with memory
//!
//! This module contains functions for querying the size and alignment of
//! types, initializing and manipulating memory.

#[allow(missing_doc)]; // FIXME

use cast;
use ptr;
use unstable::intrinsics;
use unstable::intrinsics::{bswap16, bswap32, bswap64};

/// Returns the size of a type in bytes.
#[inline]
pub fn size_of<T>() -> uint {
    unsafe { intrinsics::size_of::<T>() }
}

/// Returns the size of the type that `_val` points to in bytes.
#[inline]
pub fn size_of_val<T>(_val: &T) -> uint {
    size_of::<T>()
}

/// Returns the size of a type in bytes, or 1 if the actual size is zero.
///
/// Useful for building structures containing variable-length arrays.
#[inline]
pub fn nonzero_size_of<T>() -> uint {
    let s = size_of::<T>();
    if s == 0 { 1 } else { s }
}

/// Returns the size in bytes of the type of the value that `_val` points to.
#[inline]
pub fn nonzero_size_of_val<T>(_val: &T) -> uint {
    nonzero_size_of::<T>()
}

/// Returns the ABI-required minimum alignment of a type
///
/// This is the alignment used for struct fields. It may be smaller
/// than the preferred alignment.
#[inline]
pub fn min_align_of<T>() -> uint {
    unsafe { intrinsics::min_align_of::<T>() }
}

/// Returns the ABI-required minimum alignment of the type of the value that
/// `_val` points to
#[inline]
pub fn min_align_of_val<T>(_val: &T) -> uint {
    min_align_of::<T>()
}

/// Returns the preferred alignment of a type
#[inline]
pub fn pref_align_of<T>() -> uint {
    unsafe { intrinsics::pref_align_of::<T>() }
}

/// Returns the preferred alignment of the type of the value that
/// `_val` points to
#[inline]
pub fn pref_align_of_val<T>(_val: &T) -> uint {
    pref_align_of::<T>()
}

/// Create a value initialized to zero.
///
/// `init` is unsafe because it returns a zeroed-out datum,
/// which is unsafe unless T is Pod.
#[inline]
pub unsafe fn init<T>() -> T {
    intrinsics::init()
}

/// Create an uninitialized value.
#[inline]
pub unsafe fn uninit<T>() -> T {
    intrinsics::uninit()
}

/// Move a value to an uninitialized memory location.
///
/// Drop glue is not run on the destination.
#[inline]
pub unsafe fn move_val_init<T>(dst: &mut T, src: T) {
    intrinsics::move_val_init(dst, src)
}

#[cfg(target_endian = "little")] #[inline] pub fn to_le16(x: i16) -> i16 { x }
#[cfg(target_endian = "big")]    #[inline] pub fn to_le16(x: i16) -> i16 { unsafe { bswap16(x) } }
#[cfg(target_endian = "little")] #[inline] pub fn to_le32(x: i32) -> i32 { x }
#[cfg(target_endian = "big")]    #[inline] pub fn to_le32(x: i32) -> i32 { unsafe { bswap32(x) } }
#[cfg(target_endian = "little")] #[inline] pub fn to_le64(x: i64) -> i64 { x }
#[cfg(target_endian = "big")]    #[inline] pub fn to_le64(x: i64) -> i64 { unsafe { bswap64(x) } }

#[cfg(target_endian = "little")] #[inline] pub fn to_be16(x: i16) -> i16 { unsafe { bswap16(x) } }
#[cfg(target_endian = "big")]    #[inline] pub fn to_be16(x: i16) -> i16 { x }
#[cfg(target_endian = "little")] #[inline] pub fn to_be32(x: i32) -> i32 { unsafe { bswap32(x) } }
#[cfg(target_endian = "big")]    #[inline] pub fn to_be32(x: i32) -> i32 { x }
#[cfg(target_endian = "little")] #[inline] pub fn to_be64(x: i64) -> i64 { unsafe { bswap64(x) } }
#[cfg(target_endian = "big")]    #[inline] pub fn to_be64(x: i64) -> i64 { x }

#[cfg(target_endian = "little")] #[inline] pub fn from_le16(x: i16) -> i16 { x }
#[cfg(target_endian = "big")]    #[inline] pub fn from_le16(x: i16) -> i16 { unsafe { bswap16(x) } }
#[cfg(target_endian = "little")] #[inline] pub fn from_le32(x: i32) -> i32 { x }
#[cfg(target_endian = "big")]    #[inline] pub fn from_le32(x: i32) -> i32 { unsafe { bswap32(x) } }
#[cfg(target_endian = "little")] #[inline] pub fn from_le64(x: i64) -> i64 { x }
#[cfg(target_endian = "big")]    #[inline] pub fn from_le64(x: i64) -> i64 { unsafe { bswap64(x) } }

#[cfg(target_endian = "little")] #[inline] pub fn from_be16(x: i16) -> i16 { unsafe { bswap16(x) } }
#[cfg(target_endian = "big")]    #[inline] pub fn from_be16(x: i16) -> i16 { x }
#[cfg(target_endian = "little")] #[inline] pub fn from_be32(x: i32) -> i32 { unsafe { bswap32(x) } }
#[cfg(target_endian = "big")]    #[inline] pub fn from_be32(x: i32) -> i32 { x }
#[cfg(target_endian = "little")] #[inline] pub fn from_be64(x: i64) -> i64 { unsafe { bswap64(x) } }
#[cfg(target_endian = "big")]    #[inline] pub fn from_be64(x: i64) -> i64 { x }


/**
 * Swap the values at two mutable locations of the same type, without
 * deinitialising or copying either one.
 */
#[inline]
pub fn swap<T>(x: &mut T, y: &mut T) {
    unsafe {
        // Give ourselves some scratch space to work with
        let mut t: T = uninit();

        // Perform the swap, `&mut` pointers never alias
        ptr::copy_nonoverlapping_memory(&mut t, &*x, 1);
        ptr::copy_nonoverlapping_memory(x, &*y, 1);
        ptr::copy_nonoverlapping_memory(y, &t, 1);

        // y and t now point to the same thing, but we need to completely forget `tmp`
        // because it's no longer relevant.
        cast::forget(t);
    }
}

/**
 * Replace the value at a mutable location with a new one, returning the old
 * value, without deinitialising or copying either one.
 */
#[inline]
pub fn replace<T>(dest: &mut T, mut src: T) -> T {
    swap(dest, &mut src);
    src
}

/// Disposes of a value.
#[inline]
pub fn drop<T>(_x: T) { }

#[cfg(test)]
mod tests {
    use mem::*;
    use option::{Some,None};

    #[test]
    fn size_of_basic() {
        assert_eq!(size_of::<u8>(), 1u);
        assert_eq!(size_of::<u16>(), 2u);
        assert_eq!(size_of::<u32>(), 4u);
        assert_eq!(size_of::<u64>(), 8u);
    }

    #[test]
    #[cfg(target_arch = "x86")]
    #[cfg(target_arch = "arm")]
    #[cfg(target_arch = "mips")]
    fn size_of_32() {
        assert_eq!(size_of::<uint>(), 4u);
        assert_eq!(size_of::<*uint>(), 4u);
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn size_of_64() {
        assert_eq!(size_of::<uint>(), 8u);
        assert_eq!(size_of::<*uint>(), 8u);
    }

    #[test]
    fn size_of_val_basic() {
        assert_eq!(size_of_val(&1u8), 1);
        assert_eq!(size_of_val(&1u16), 2);
        assert_eq!(size_of_val(&1u32), 4);
        assert_eq!(size_of_val(&1u64), 8);
    }

    #[test]
    fn nonzero_size_of_basic() {
        type Z = [i8, ..0];
        assert_eq!(size_of::<Z>(), 0u);
        assert_eq!(nonzero_size_of::<Z>(), 1u);
        assert_eq!(nonzero_size_of::<uint>(), size_of::<uint>());
    }

    #[test]
    fn nonzero_size_of_val_basic() {
        let z = [0u8, ..0];
        assert_eq!(size_of_val(&z), 0u);
        assert_eq!(nonzero_size_of_val(&z), 1u);
        assert_eq!(nonzero_size_of_val(&1u), size_of_val(&1u));
    }

    #[test]
    fn align_of_basic() {
        assert_eq!(pref_align_of::<u8>(), 1u);
        assert_eq!(pref_align_of::<u16>(), 2u);
        assert_eq!(pref_align_of::<u32>(), 4u);
    }

    #[test]
    #[cfg(target_arch = "x86")]
    #[cfg(target_arch = "arm")]
    #[cfg(target_arch = "mips")]
    fn align_of_32() {
        assert_eq!(pref_align_of::<uint>(), 4u);
        assert_eq!(pref_align_of::<*uint>(), 4u);
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn align_of_64() {
        assert_eq!(pref_align_of::<uint>(), 8u);
        assert_eq!(pref_align_of::<*uint>(), 8u);
    }

    #[test]
    fn align_of_val_basic() {
        assert_eq!(pref_align_of_val(&1u8), 1u);
        assert_eq!(pref_align_of_val(&1u16), 2u);
        assert_eq!(pref_align_of_val(&1u32), 4u);
    }

    #[test]
    fn test_swap() {
        let mut x = 31337;
        let mut y = 42;
        swap(&mut x, &mut y);
        assert_eq!(x, 42);
        assert_eq!(y, 31337);
    }

    #[test]
    fn test_replace() {
        let mut x = Some(~"test");
        let y = replace(&mut x, None);
        assert!(x.is_none());
        assert!(y.is_some());
    }
}

/// Completely miscellaneous language-construct benchmarks.
#[cfg(test)]
mod bench {
    extern crate test;
    use self::test::BenchHarness;
    use option::{Some,None};

    // Static/dynamic method dispatch

    struct Struct {
        field: int
    }

    trait Trait {
        fn method(&self) -> int;
    }

    impl Trait for Struct {
        fn method(&self) -> int {
            self.field
        }
    }

    #[bench]
    fn trait_vtable_method_call(bh: &mut BenchHarness) {
        let s = Struct { field: 10 };
        let t = &s as &Trait;
        bh.iter(|| {
            t.method()
        });
    }

    #[bench]
    fn trait_static_method_call(bh: &mut BenchHarness) {
        let s = Struct { field: 10 };
        bh.iter(|| {
            s.method()
        });
    }

    // Overhead of various match forms

    #[bench]
    fn match_option_some(bh: &mut BenchHarness) {
        let x = Some(10);
        bh.iter(|| {
            match x {
                Some(y) => y,
                None => 11
            }
        });
    }

    #[bench]
    fn match_vec_pattern(bh: &mut BenchHarness) {
        let x = [1,2,3,4,5,6];
        bh.iter(|| {
            match x {
                [1,2,3,..] => 10,
                _ => 11
            }
        });
    }
}
