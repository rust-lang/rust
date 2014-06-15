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

use clone::Clone;
use ptr;
use intrinsics;
use intrinsics::{bswap16, bswap32, bswap64};

pub use intrinsics::transmute;

/// Returns the size of a type in bytes.
#[inline]
#[stable]
pub fn size_of<T>() -> uint {
    unsafe { intrinsics::size_of::<T>() }
}

/// Returns the size of the type that `_val` points to in bytes.
#[inline]
#[stable]
pub fn size_of_val<T>(_val: &T) -> uint {
    size_of::<T>()
}

/// Deprecated, this function will be removed soon
#[inline]
#[deprecated = "this function will be removed soon"]
pub fn nonzero_size_of<T>() -> uint {
    match size_of::<T>() {
        0 => 1,
        n => n,
    }
}

/// Deprecated, this function will be removed soon
#[inline]
#[deprecated = "this function will be removed soon"]
pub fn nonzero_size_of_val<T>(val: &T) -> uint {
    match size_of_val::<T>(val) {
        0 => 1,
        n => n,
    }
}

/// Returns the ABI-required minimum alignment of a type
///
/// This is the alignment used for struct fields. It may be smaller
/// than the preferred alignment.
#[inline]
#[stable]
pub fn min_align_of<T>() -> uint {
    unsafe { intrinsics::min_align_of::<T>() }
}

/// Returns the ABI-required minimum alignment of the type of the value that
/// `_val` points to
#[inline]
#[stable]
pub fn min_align_of_val<T>(_val: &T) -> uint {
    min_align_of::<T>()
}

/// Returns the alignment in memory for a type.
///
/// This function will return the alignment, in bytes, of a type in memory. If
/// the alignment returned is adhered to, then the type is guaranteed to
/// function properly.
#[inline]
#[stable]
pub fn align_of<T>() -> uint {
    // We use the preferred alignment as the default alignment for a type. This
    // appears to be what clang migrated towards as well:
    //
    // http://lists.cs.uiuc.edu/pipermail/cfe-commits/Week-of-Mon-20110725/044411.html
    unsafe { intrinsics::pref_align_of::<T>() }
}

/// Returns the alignment of the type of the value that `_val` points to.
///
/// This is similar to `align_of`, but function will properly handle types such
/// as trait objects (in the future), returning the alignment for an arbitrary
/// value at runtime.
#[inline]
#[stable]
pub fn align_of_val<T>(_val: &T) -> uint {
    align_of::<T>()
}

/// Deprecated, this function has been renamed to align_of
#[inline]
#[deprecated = "use mem::align_of instead"]
pub fn pref_align_of<T>() -> uint { align_of::<T>() }

/// Deprecated, this function has been renamed to align_of_val
#[inline]
#[deprecated = "use mem::align_of_val instead"]
pub fn pref_align_of_val<T>(val: &T) -> uint { align_of_val(val) }

/// Create a value initialized to zero.
///
/// This function is similar to allocating space for a a local variable and
/// zeroing it out (an unsafe operation).
///
/// Care must be taken when using this function, if the type `T` has a
/// destructor and the value falls out of scope (due to unwinding or returning)
/// before being initialized, then the destructor will run on zeroed
/// data, likely leading to crashes.
///
/// This is useful for FFI functions sometimes, but should generally be avoided.
#[inline]
#[stable]
pub unsafe fn zeroed<T>() -> T {
    intrinsics::init()
}

/// Deprecated, use zeroed() instead
#[inline]
#[deprecated = "this function has been renamed to zeroed()"]
pub unsafe fn init<T>() -> T { zeroed() }

/// Create an uninitialized value.
///
/// Care must be taken when using this function, if the type `T` has a
/// destructor and the value falls out of scope (due to unwinding or returning)
/// before being initialized, then the destructor will run on uninitialized
/// data, likely leading to crashes.
///
/// This is useful for FFI functions sometimes, but should generally be avoided.
#[inline]
#[stable]
pub unsafe fn uninitialized<T>() -> T {
    intrinsics::uninit()
}

/// Deprecated, use `uninitialized` instead.
#[inline]
#[deprecated = "this function has been renamed to `uninitialized`"]
pub unsafe fn uninit<T>() -> T {
    intrinsics::uninit()
}

/// Unsafely overwrite a memory location with the given value without destroying
/// the old value.
///
/// This operation is unsafe because it does not destroy the previous value
/// contained at the location `dst`. This could leak allocations or resources,
/// so care must be taken to previously deallocate the value at `dst`.
#[inline]
#[deprecated = "use ptr::write"]
pub unsafe fn overwrite<T>(dst: *mut T, src: T) {
    intrinsics::move_val_init(&mut *dst, src)
}

/// Deprecated, use `overwrite` instead
#[inline]
#[deprecated = "use ptr::write"]
pub unsafe fn move_val_init<T>(dst: &mut T, src: T) {
    ptr::write(dst, src)
}

/// A type that can have its bytes re-ordered.
pub trait ByteOrder: Clone {
    /// Reverses the byte order of the value.
    ///
    /// # Example
    ///
    /// ```rust
    /// use std::mem::ByteOrder;
    ///
    /// let n = 0x0123456789ABCDEFu64;
    /// let m = 0xEFCDAB8967452301u64;
    ///
    /// assert_eq!(n.swap_bytes(), m);
    /// ```
    fn swap_bytes(&self) -> Self;

    /// Convert a value from big endian to the target's endianness.
    ///
    /// On big endian this is a no-op. On little endian the bytes are swapped.
    ///
    /// # Example
    ///
    /// ```rust
    /// use std::mem::ByteOrder;
    ///
    /// let n = 0x0123456789ABCDEFu64;
    ///
    /// if cfg!(target_endian = "big") {
    ///     assert_eq!(ByteOrder::from_big_endian(n), n)
    /// } else {
    ///     assert_eq!(ByteOrder::from_big_endian(n), n.swap_bytes())
    /// }
    /// ```
    #[inline]
    fn from_big_endian(x: Self) -> Self {
        if cfg!(target_endian = "big") { x } else { x.swap_bytes() }
    }

    /// Convert a value from little endian to the target's endianness.
    ///
    /// On little endian this is a no-op. On big endian the bytes are swapped.
    ///
    /// # Example
    ///
    /// ```rust
    /// use std::mem::ByteOrder;
    ///
    /// let n = 0x0123456789ABCDEFu64;
    ///
    /// if cfg!(target_endian = "little") {
    ///     assert_eq!(ByteOrder::from_little_endian(n), n)
    /// } else {
    ///     assert_eq!(ByteOrder::from_little_endian(n), n.swap_bytes())
    /// }
    /// ```
    #[inline]
    fn from_little_endian(x: Self) -> Self {
        if cfg!(target_endian = "little") { x } else { x.swap_bytes() }
    }

    /// Convert the value to big endian from the target's endianness.
    ///
    /// On big endian this is a no-op. On little endian the bytes are swapped.
    ///
    /// # Example
    ///
    /// ```rust
    /// use std::mem::ByteOrder;
    ///
    /// let n = 0x0123456789ABCDEFu64;
    ///
    /// if cfg!(target_endian = "big") {
    ///     assert_eq!(n.to_big_endian(), n)
    /// } else {
    ///     assert_eq!(n.to_big_endian(), n.swap_bytes())
    /// }
    /// ```
    #[inline]
    fn to_big_endian(&self) -> Self {
        if cfg!(target_endian = "big") { self.clone() } else { self.swap_bytes() }
    }

    /// Convert the value to little endian from the target's endianness.
    ///
    /// On little endian this is a no-op. On big endian the bytes are swapped.
    ///
    /// # Example
    ///
    /// ```rust
    /// use std::mem::ByteOrder;
    ///
    /// let n = 0x0123456789ABCDEFu64;
    ///
    /// if cfg!(target_endian = "little") {
    ///     assert_eq!(n.to_little_endian(), n)
    /// } else {
    ///     assert_eq!(n.to_little_endian(), n.swap_bytes())
    /// }
    /// ```
    #[inline]
    fn to_little_endian(&self) -> Self {
        if cfg!(target_endian = "little") { self.clone() } else { self.swap_bytes() }
    }
}

impl ByteOrder for u8 {
    #[inline]
    fn swap_bytes(&self) -> u8 {
        *self // swapping a single byte does nothing
    }
}

impl ByteOrder for u16 {
    #[inline]
    fn swap_bytes(&self) -> u16 {
        unsafe { intrinsics::bswap16(*self) }
    }
}

impl ByteOrder for u32 {
    #[inline]
    fn swap_bytes(&self) -> u32 {
        unsafe { intrinsics::bswap32(*self) }
    }
}

impl ByteOrder for u64 {
    #[inline]
    fn swap_bytes(&self) -> u64 {
        unsafe { intrinsics::bswap64(*self) }
    }
}

#[cfg(target_word_size = "32")]
impl ByteOrder for uint {
    #[inline]
    fn swap_bytes(&self) -> uint {
        (*self as u32).swap_bytes() as uint
    }
}

#[cfg(target_word_size = "64")]
impl ByteOrder for uint {
    #[inline]
    fn swap_bytes(&self) -> uint {
        (*self as u64).swap_bytes() as uint
    }
}

/// Convert an u16 to little endian from the target's endianness.
///
/// On little endian, this is a no-op.  On big endian, the bytes are swapped.
#[inline]
#[stable]
pub fn to_le16(x: u16) -> u16 { x.to_little_endian() }

/// Convert an u32 to little endian from the target's endianness.
///
/// On little endian, this is a no-op.  On big endian, the bytes are swapped.
#[inline]
#[stable]
pub fn to_le32(x: u32) -> u32 { x.to_little_endian() }

/// Convert an u64 to little endian from the target's endianness.
///
/// On little endian, this is a no-op.  On big endian, the bytes are swapped.
#[inline]
#[stable]
pub fn to_le64(x: u64) -> u64 { x.to_little_endian() }

/// Convert an u16 to big endian from the target's endianness.
///
/// On big endian, this is a no-op.  On little endian, the bytes are swapped.
#[inline]
#[stable]
pub fn to_be16(x: u16) -> u16 { x.to_big_endian() }

/// Convert an u32 to big endian from the target's endianness.
///
/// On big endian, this is a no-op.  On little endian, the bytes are swapped.
#[inline]
#[stable]
pub fn to_be32(x: u32) -> u32 { x.to_big_endian() }

/// Convert an u64 to big endian from the target's endianness.
///
/// On big endian, this is a no-op.  On little endian, the bytes are swapped.
#[inline]
#[stable]
pub fn to_be64(x: u64) -> u64 { x.to_big_endian() }

/// Convert an u16 from little endian to the target's endianness.
///
/// On little endian, this is a no-op.  On big endian, the bytes are swapped.
#[inline]
#[stable]
pub fn from_le16(x: u16) -> u16 { ByteOrder::from_little_endian(x) }

/// Convert an u32 from little endian to the target's endianness.
///
/// On little endian, this is a no-op.  On big endian, the bytes are swapped.
#[inline]
#[stable]
pub fn from_le32(x: u32) -> u32 { ByteOrder::from_little_endian(x) }

/// Convert an u64 from little endian to the target's endianness.
///
/// On little endian, this is a no-op.  On big endian, the bytes are swapped.
#[inline]
#[stable]
pub fn from_le64(x: u64) -> u64 { ByteOrder::from_little_endian(x) }

/// Convert an u16 from big endian to the target's endianness.
///
/// On big endian, this is a no-op.  On little endian, the bytes are swapped.
#[inline]
#[stable]
pub fn from_be16(x: u16) -> u16 { ByteOrder::from_big_endian(x) }

/// Convert an u32 from big endian to the target's endianness.
///
/// On big endian, this is a no-op.  On little endian, the bytes are swapped.
#[inline]
#[stable]
pub fn from_be32(x: u32) -> u32 { ByteOrder::from_big_endian(x) }

/// Convert an u64 from big endian to the target's endianness.
///
/// On big endian, this is a no-op.  On little endian, the bytes are swapped.
#[inline]
#[stable]
pub fn from_be64(x: u64) -> u64 { ByteOrder::from_big_endian(x) }

/**
 * Swap the values at two mutable locations of the same type, without
 * deinitialising or copying either one.
 */
#[inline]
#[stable]
pub fn swap<T>(x: &mut T, y: &mut T) {
    unsafe {
        // Give ourselves some scratch space to work with
        let mut t: T = uninitialized();

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
#[stable]
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
#[stable]
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
    use realstd::owned::Box;
    use realstd::vec::Vec;
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
    fn align_of_basic() {
        assert_eq!(align_of::<u8>(), 1u);
        assert_eq!(align_of::<u16>(), 2u);
        assert_eq!(align_of::<u32>(), 4u);
    }

    #[test]
    #[cfg(target_arch = "x86")]
    #[cfg(target_arch = "arm")]
    #[cfg(target_arch = "mips")]
    fn align_of_32() {
        assert_eq!(align_of::<uint>(), 4u);
        assert_eq!(align_of::<*uint>(), 4u);
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn align_of_64() {
        assert_eq!(align_of::<uint>(), 8u);
        assert_eq!(align_of::<*uint>(), 8u);
    }

    #[test]
    fn align_of_val_basic() {
        assert_eq!(align_of_val(&1u8), 1u);
        assert_eq!(align_of_val(&1u16), 2u);
        assert_eq!(align_of_val(&1u32), 4u);
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
        let mut x = Some("test".to_string());
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
            assert!(Vec::from_slice([76u8]) == transmute("L".to_string()));
        }
    }

    macro_rules! test_byte_order {
        ($T:ident) => {
            mod $T {
                use mem::ByteOrder;

                static A: $T = 0b0101100;
                static B: $T = 0b0100001;
                static C: $T = 0b1111001;

                static _0: $T = 0;
                static _1: $T = !0;

                #[test]
                fn test_swap_bytes() {
                    assert_eq!(A.swap_bytes().swap_bytes(), A);
                    assert_eq!(B.swap_bytes().swap_bytes(), B);
                    assert_eq!(C.swap_bytes().swap_bytes(), C);

                    // Swapping these should make no difference
                    assert_eq!(_0.swap_bytes(), _0);
                    assert_eq!(_1.swap_bytes(), _1);
                }

                #[test]
                fn test_little_endian() {
                    assert_eq!(ByteOrder::from_little_endian(A.to_little_endian()), A);
                    assert_eq!(ByteOrder::from_little_endian(B.to_little_endian()), B);
                    assert_eq!(ByteOrder::from_little_endian(C.to_little_endian()), C);
                    assert_eq!(ByteOrder::from_little_endian(_0), _0);
                    assert_eq!(ByteOrder::from_little_endian(_1), _1);
                    assert_eq!(_0.to_little_endian(), _0);
                    assert_eq!(_1.to_little_endian(), _1);
                }

                #[test]
                fn test_big_endian() {
                    assert_eq!(ByteOrder::from_big_endian(A.to_big_endian()), A);
                    assert_eq!(ByteOrder::from_big_endian(B.to_big_endian()), B);
                    assert_eq!(ByteOrder::from_big_endian(C.to_big_endian()), C);
                    assert_eq!(ByteOrder::from_big_endian(_0), _0);
                    assert_eq!(ByteOrder::from_big_endian(_1), _1);
                    assert_eq!(_0.to_big_endian(), _0);
                    assert_eq!(_1.to_big_endian(), _1);
                }
            }
        }
    }

    test_byte_order!(u8)
    test_byte_order!(u16)
    test_byte_order!(u32)
    test_byte_order!(u64)
    test_byte_order!(uint)
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
