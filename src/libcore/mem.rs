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

use ptr;
use intrinsics;
use intrinsics::{bswap16, bswap32, bswap64};

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
    match size_of::<T>() {
        0 => 1,
        x => x
    }
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
/// which is unsafe unless T is Copy.
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

/// Convert an u16 to little endian from the target's endianness.
///
/// On little endian, this is a no-op.  On big endian, the bytes are swapped.
#[cfg(target_endian = "little")] #[inline] pub fn to_le16(x: u16) -> u16 { x }

/// Convert an u16 to little endian from the target's endianness.
///
/// On little endian, this is a no-op.  On big endian, the bytes are swapped.
#[cfg(target_endian = "big")]    #[inline] pub fn to_le16(x: u16) -> u16 { unsafe { bswap16(x) } }

/// Convert an u32 to little endian from the target's endianness.
///
/// On little endian, this is a no-op.  On big endian, the bytes are swapped.
#[cfg(target_endian = "little")] #[inline] pub fn to_le32(x: u32) -> u32 { x }

/// Convert an u32 to little endian from the target's endianness.
///
/// On little endian, this is a no-op.  On big endian, the bytes are swapped.
#[cfg(target_endian = "big")]    #[inline] pub fn to_le32(x: u32) -> u32 { unsafe { bswap32(x) } }

/// Convert an u64 to little endian from the target's endianness.
///
/// On little endian, this is a no-op.  On big endian, the bytes are swapped.
#[cfg(target_endian = "little")] #[inline] pub fn to_le64(x: u64) -> u64 { x }

/// Convert an u64 to little endian from the target's endianness.
///
/// On little endian, this is a no-op.  On big endian, the bytes are swapped.
#[cfg(target_endian = "big")]    #[inline] pub fn to_le64(x: u64) -> u64 { unsafe { bswap64(x) } }


/// Convert an u16 to big endian from the target's endianness.
///
/// On big endian, this is a no-op.  On little endian, the bytes are swapped.
#[cfg(target_endian = "little")] #[inline] pub fn to_be16(x: u16) -> u16 { unsafe { bswap16(x) } }

/// Convert an u16 to big endian from the target's endianness.
///
/// On big endian, this is a no-op.  On little endian, the bytes are swapped.
#[cfg(target_endian = "big")]    #[inline] pub fn to_be16(x: u16) -> u16 { x }

/// Convert an u32 to big endian from the target's endianness.
///
/// On big endian, this is a no-op.  On little endian, the bytes are swapped.
#[cfg(target_endian = "little")] #[inline] pub fn to_be32(x: u32) -> u32 { unsafe { bswap32(x) } }

/// Convert an u32 to big endian from the target's endianness.
///
/// On big endian, this is a no-op.  On little endian, the bytes are swapped.
#[cfg(target_endian = "big")]    #[inline] pub fn to_be32(x: u32) -> u32 { x }

/// Convert an u64 to big endian from the target's endianness.
///
/// On big endian, this is a no-op.  On little endian, the bytes are swapped.
#[cfg(target_endian = "little")] #[inline] pub fn to_be64(x: u64) -> u64 { unsafe { bswap64(x) } }

/// Convert an u64 to big endian from the target's endianness.
///
/// On big endian, this is a no-op.  On little endian, the bytes are swapped.
#[cfg(target_endian = "big")]    #[inline] pub fn to_be64(x: u64) -> u64 { x }


/// Convert an u16 from little endian to the target's endianness.
///
/// On little endian, this is a no-op.  On big endian, the bytes are swapped.
#[cfg(target_endian = "little")] #[inline] pub fn from_le16(x: u16) -> u16 { x }

/// Convert an u16 from little endian to the target's endianness.
///
/// On little endian, this is a no-op.  On big endian, the bytes are swapped.
#[cfg(target_endian = "big")]    #[inline] pub fn from_le16(x: u16) -> u16 { unsafe { bswap16(x) } }

/// Convert an u32 from little endian to the target's endianness.
///
/// On little endian, this is a no-op.  On big endian, the bytes are swapped.
#[cfg(target_endian = "little")] #[inline] pub fn from_le32(x: u32) -> u32 { x }

/// Convert an u32 from little endian to the target's endianness.
///
/// On little endian, this is a no-op.  On big endian, the bytes are swapped.
#[cfg(target_endian = "big")]    #[inline] pub fn from_le32(x: u32) -> u32 { unsafe { bswap32(x) } }

/// Convert an u64 from little endian to the target's endianness.
///
/// On little endian, this is a no-op.  On big endian, the bytes are swapped.
#[cfg(target_endian = "little")] #[inline] pub fn from_le64(x: u64) -> u64 { x }

/// Convert an u64 from little endian to the target's endianness.
///
/// On little endian, this is a no-op.  On big endian, the bytes are swapped.
#[cfg(target_endian = "big")]    #[inline] pub fn from_le64(x: u64) -> u64 { unsafe { bswap64(x) } }


/// Convert an u16 from big endian to the target's endianness.
///
/// On big endian, this is a no-op.  On little endian, the bytes are swapped.
#[cfg(target_endian = "little")] #[inline] pub fn from_be16(x: u16) -> u16 { unsafe { bswap16(x) } }

/// Convert an u16 from big endian to the target's endianness.
///
/// On big endian, this is a no-op.  On little endian, the bytes are swapped.
#[cfg(target_endian = "big")]    #[inline] pub fn from_be16(x: u16) -> u16 { x }

/// Convert an u32 from big endian to the target's endianness.
///
/// On big endian, this is a no-op.  On little endian, the bytes are swapped.
#[cfg(target_endian = "little")] #[inline] pub fn from_be32(x: u32) -> u32 { unsafe { bswap32(x) } }

/// Convert an u32 from big endian to the target's endianness.
///
/// On big endian, this is a no-op.  On little endian, the bytes are swapped.
#[cfg(target_endian = "big")]    #[inline] pub fn from_be32(x: u32) -> u32 { x }

/// Convert an u64 from big endian to the target's endianness.
///
/// On big endian, this is a no-op.  On little endian, the bytes are swapped.
#[cfg(target_endian = "little")] #[inline] pub fn from_be64(x: u64) -> u64 { unsafe { bswap64(x) } }

/// Convert an u64 from big endian to the target's endianness.
///
/// On big endian, this is a no-op.  On little endian, the bytes are swapped.
#[cfg(target_endian = "big")]    #[inline] pub fn from_be64(x: u64) -> u64 { x }

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

        // y and t now point to the same thing, but we need to completely forget `t`
        // because it's no longer relevant.
        forget(t);
    }
}

/**
 * Replace the value at a mutable location with a new one, returning the old
 * value, without deinitialising or copying either one.
 *
 * This is primarily used for transferring and swapping ownership of a value
 * in a mutable location. For example, this function allows consumption of
 * one field of a struct by replacing it with another value. The normal approach
 * doesn't always work:
 *
 * ```rust,ignore
 * struct Buffer<T> { buf: Vec<T> }
 *
 * impl<T> Buffer<T> {
 *     fn get_and_reset(&mut self) -> Vec<T> {
 *         // error: cannot move out of dereference of `&mut`-pointer
 *         let buf = self.buf;
 *         self.buf = Vec::new();
 *         buf
 *     }
 * }
 * ```
 *
 * Note that `T` does not necessarily implement `Clone`, so it can't even
 * clone and reset `self.buf`. But `replace` can be used to disassociate
 * the original value of `self.buf` from `self`, allowing it to be returned:
 *
 * ```rust
 * # struct Buffer<T> { buf: Vec<T> }
 * impl<T> Buffer<T> {
 *     fn get_and_reset(&mut self) -> Vec<T> {
 *         use std::mem::replace;
 *         replace(&mut self.buf, Vec::new())
 *     }
 * }
 * ```
 */
#[inline]
pub fn replace<T>(dest: &mut T, mut src: T) -> T {
    swap(dest, &mut src);
    src
}

/// Disposes of a value.
///
/// This function can be used to destroy any value by allowing `drop` to take
/// ownership of its argument.
///
/// # Example
///
/// ```
/// use std::cell::RefCell;
///
/// let x = RefCell::new(1);
///
/// let mut mutable_borrow = x.borrow_mut();
/// *mutable_borrow = 1;
/// drop(mutable_borrow); // relinquish the mutable borrow on this slot
///
/// let borrow = x.borrow();
/// println!("{}", *borrow);
/// ```
#[inline]
pub fn drop<T>(_x: T) { }

/// Moves a thing into the void.
///
/// The forget function will take ownership of the provided value but neglect
/// to run any required cleanup or memory management operations on it.
///
/// This function is the unsafe version of the `drop` function because it does
/// not run any destructors.
#[inline]
#[stable]
pub unsafe fn forget<T>(thing: T) { intrinsics::forget(thing) }

/// Unsafely transforms a value of one type into a value of another type.
///
/// Both types must have the same size and alignment, and this guarantee is
/// enforced at compile-time.
///
/// # Example
///
/// ```rust
/// use std::mem;
///
/// let v: &[u8] = unsafe { mem::transmute("L") };
/// assert!(v == [76u8]);
/// ```
#[inline]
#[unstable = "this function will be modified to reject invocations of it which \
              cannot statically prove that T and U are the same size. For \
              example, this function, as written today, will be rejected in \
              the future because the size of T and U cannot be statically \
              known to be the same"]
pub unsafe fn transmute<T, U>(thing: T) -> U {
    intrinsics::transmute(thing)
}

/// Interprets `src` as `&U`, and then reads `src` without moving the contained
/// value.
///
/// This function will unsafely assume the pointer `src` is valid for
/// `sizeof(U)` bytes by transmuting `&T` to `&U` and then reading the `&U`. It
/// will also unsafely create a copy of the contained value instead of moving
/// out of `src`.
///
/// It is not a compile-time error if `T` and `U` have different sizes, but it
/// is highly encouraged to only invoke this function where `T` and `U` have the
/// same size. This function triggers undefined behavior if `U` is larger than
/// `T`.
#[inline]
#[stable]
pub unsafe fn transmute_copy<T, U>(src: &T) -> U {
    ptr::read(src as *T as *U)
}

/// Transforms lifetime of the second pointer to match the first.
#[inline]
#[unstable = "this function may be removed in the future due to its \
              questionable utility"]
pub unsafe fn copy_lifetime<'a, S, T>(_ptr: &'a S, ptr: &T) -> &'a T {
    transmute(ptr)
}

/// Transforms lifetime of the second mutable pointer to match the first.
#[inline]
#[unstable = "this function may be removed in the future due to its \
              questionable utility"]
pub unsafe fn copy_mut_lifetime<'a, S, T>(_ptr: &'a mut S,
                                          ptr: &mut T) -> &'a mut T {
    transmute(ptr)
}

#[cfg(test)]
mod tests {
    use mem::*;
    use option::{Some,None};
    use realstd::str::StrAllocating;
    use owned::Box;
    use raw;

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
        let mut x = Some("test".to_owned());
        let y = replace(&mut x, None);
        assert!(x.is_none());
        assert!(y.is_some());
    }

    #[test]
    fn test_transmute_copy() {
        assert_eq!(1u, unsafe { ::mem::transmute_copy(&1) });
    }

    #[test]
    fn test_transmute() {
        trait Foo {}
        impl Foo for int {}

        let a = box 100 as Box<Foo>;
        unsafe {
            let x: raw::TraitObject = transmute(a);
            assert!(*(x.data as *int) == 100);
            let _x: Box<Foo> = transmute(x);
        }

        unsafe {
            assert_eq!(box [76u8], transmute("L".to_owned()));
        }
    }
}

// FIXME #13642 (these benchmarks should be in another place)
/// Completely miscellaneous language-construct benchmarks.
#[cfg(test)]
mod bench {
    extern crate test;
    use self::test::Bencher;
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
    fn trait_vtable_method_call(b: &mut Bencher) {
        let s = Struct { field: 10 };
        let t = &s as &Trait;
        b.iter(|| {
            t.method()
        });
    }

    #[bench]
    fn trait_static_method_call(b: &mut Bencher) {
        let s = Struct { field: 10 };
        b.iter(|| {
            s.method()
        });
    }

    // Overhead of various match forms

    #[bench]
    fn match_option_some(b: &mut Bencher) {
        let x = Some(10);
        b.iter(|| {
            match x {
                Some(y) => y,
                None => 11
            }
        });
    }

    #[bench]
    fn match_vec_pattern(b: &mut Bencher) {
        let x = [1,2,3,4,5,6];
        b.iter(|| {
            match x {
                [1,2,3,..] => 10,
                _ => 11
            }
        });
    }
}
