// Copyright 2012-2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Unsafe pointer utility functions

use intrinsics;

/// Calculate the offset from a pointer.
/// The `count` argument is in units of T; e.g. a `count` of 3
/// represents a pointer offset of `3 * sizeof::<T>()` bytes.
#[inline]
pub unsafe fn offset<T>(ptr: *T, count: int) -> *T {
    intrinsics::offset(ptr, count)
}

/// Calculate the offset from a mut pointer. The count *must* be in bounds or
/// otherwise the loads of this address are undefined.
/// The `count` argument is in units of T; e.g. a `count` of 3
/// represents a pointer offset of `3 * sizeof::<T>()` bytes.
#[inline]
pub unsafe fn mut_offset<T>(ptr: *mut T, count: int) -> *mut T {
    intrinsics::offset(ptr as *T, count) as *mut T
}

/// Return the offset of the first null pointer in `buf`.
#[inline]
pub unsafe fn buf_len<T>(buf: **T) -> uint {
    position(buf, |i| *i == null())
}
/// Return the first offset `i` such that `f(buf[i]) == true`.
#[inline]
pub unsafe fn position<T>(buf: *T, f: |&T| -> bool) -> uint {
    let mut i = 0;
    loop {
        if f(&(*offset(buf, i as int))) { return i; }
        else { i += 1; }
    }
}

/// Create an unsafe null pointer
#[inline]
pub fn null<T>() -> *T { 0 as *T }

/// Create an unsafe mutable null pointer
#[inline]
pub fn mut_null<T>() -> *mut T { 0 as *mut T }

/// Returns true if the pointer is equal to the null pointer.
#[inline]
pub fn is_null<T,P:RawPtr<T>>(ptr: P) -> bool { ptr.is_null() }

/// Returns true if the pointer is not equal to the null pointer.
#[inline]
pub fn is_not_null<T,P:RawPtr<T>>(ptr: P) -> bool { ptr.is_not_null() }

/**
 * Copies data from one location to another.
 *
 * Copies `count` elements (not bytes) from `src` to `dst`. The source
 * and destination may overlap.
 */
#[inline]
pub unsafe fn copy_memory<T>(dst: *mut T, src: *T, count: uint) {
    intrinsics::copy_memory(dst, src, count)
}

/**
 * Copies data from one location to another.
 *
 * Copies `count` elements (not bytes) from `src` to `dst`. The source
 * and destination may *not* overlap.
 */
#[inline]
pub unsafe fn copy_nonoverlapping_memory<T>(dst: *mut T,
                                            src: *T,
                                            count: uint) {
    intrinsics::copy_nonoverlapping_memory(dst, src, count)
}

/**
 * Invokes memset on the specified pointer, setting `count * size_of::<T>()`
 * bytes of memory starting at `dst` to `c`.
 */
#[inline]
pub unsafe fn set_memory<T>(dst: *mut T, c: u8, count: uint) {
    intrinsics::set_memory(dst, c, count)
}

/**
 * Zeroes out `count * size_of::<T>` bytes of memory at `dst`
 */
#[inline]
pub unsafe fn zero_memory<T>(dst: *mut T, count: uint) {
    set_memory(dst, 0, count);
}

/**
 * Swap the values at two mutable locations of the same type, without
 * deinitialising or copying either one.
 */
#[inline]
pub unsafe fn swap_ptr<T>(x: *mut T, y: *mut T) {
    // Give ourselves some scratch space to work with
    let mut tmp: T = intrinsics::uninit();
    let t: *mut T = &mut tmp;

    // Perform the swap
    copy_nonoverlapping_memory(t, &*x, 1);
    copy_memory(x, &*y, 1); // `x` and `y` may overlap
    copy_nonoverlapping_memory(y, &*t, 1);

    // y and t now point to the same thing, but we need to completely forget `tmp`
    // because it's no longer relevant.
    intrinsics::forget(tmp);
}

/**
 * Replace the value at a mutable location with a new one, returning the old
 * value, without deinitialising or copying either one.
 */
#[inline]
pub unsafe fn replace_ptr<T>(dest: *mut T, mut src: T) -> T {
    swap_ptr(dest, &mut src);
    src
}

/**
 * Reads the value from `*src` and returns it. Does not copy `*src`.
 */
#[inline(always)]
pub unsafe fn read_ptr<T>(src: *T) -> T {
    let mut tmp: T = intrinsics::uninit();
    copy_nonoverlapping_memory(&mut tmp, src, 1);
    tmp
}

/**
 * Reads the value from `*src` and nulls it out.
 * This currently prevents destructors from executing.
 */
#[inline(always)]
pub unsafe fn read_and_zero_ptr<T>(dest: *mut T) -> T {
    // Copy the data out from `dest`:
    let tmp = read_ptr(&*dest);

    // Now zero out `dest`:
    zero_memory(dest, 1);

    tmp
}

/// Transform a region pointer - &T - to an unsafe pointer - *T.
#[inline]
pub fn to_unsafe_ptr<T>(thing: &T) -> *T {
    thing as *T
}

/// Transform a mutable region pointer - &mut T - to a mutable unsafe pointer - *mut T.
#[inline]
pub fn to_mut_unsafe_ptr<T>(thing: &mut T) -> *mut T {
    thing as *mut T
}

#[allow(missing_doc)]
pub trait RawPtr<T> {
    fn null() -> Self;
    fn is_null(&self) -> bool;
    fn is_not_null(&self) -> bool;
    fn to_uint(&self) -> uint;
    unsafe fn offset(self, count: int) -> Self;
}

/// Extension methods for immutable pointers
impl<T> RawPtr<T> for *T {
    /// Returns the null pointer.
    #[inline]
    fn null() -> *T { null() }

    /// Returns true if the pointer is equal to the null pointer.
    #[inline]
    fn is_null(&self) -> bool { *self == RawPtr::null() }

    /// Returns true if the pointer is not equal to the null pointer.
    #[inline]
    fn is_not_null(&self) -> bool { *self != RawPtr::null() }

    /// Returns the address of this pointer.
    #[inline]
    fn to_uint(&self) -> uint { *self as uint }

    /// Calculates the offset from a pointer. The offset *must* be in-bounds of
    /// the object, or one-byte-past-the-end.
    #[inline]
    unsafe fn offset(self, count: int) -> *T { offset(self, count) }
}

/// Extension methods for mutable pointers
impl<T> RawPtr<T> for *mut T {
    /// Returns the null pointer.
    #[inline]
    fn null() -> *mut T { mut_null() }

    /// Returns true if the pointer is equal to the null pointer.
    #[inline]
    fn is_null(&self) -> bool { *self == RawPtr::null() }

    /// Returns true if the pointer is not equal to the null pointer.
    #[inline]
    fn is_not_null(&self) -> bool { *self != RawPtr::null() }

    /// Returns the address of this pointer.
    #[inline]
    fn to_uint(&self) -> uint { *self as uint }

    /// Calculates the offset from a pointer. The offset *must* be in-bounds of
    /// the object, or one-byte-past-the-end. An arithmetic overflow is also
    /// undefined behaviour.
    ///
    /// This method should be preferred over `offset` when the guarantee can be
    /// satisfied, to enable better optimization.
    #[inline]
    unsafe fn offset(self, count: int) -> *mut T { mut_offset(self, count) }
}

#[cfg(test)]
pub mod ptr_tests {
    use super::*;
    use std::prelude::*;

    use std::c_str::ToCStr;
    use std::cast;
    use std::libc;
    use std::vec::{ImmutableVector, MutableVector};

    #[test]
    fn test() {
        unsafe {
            struct Pair {
                fst: int,
                snd: int
            };
            let mut p = Pair {fst: 10, snd: 20};
            let pptr: *mut Pair = &mut p;
            let iptr: *mut int = cast::transmute(pptr);
            assert_eq!(*iptr, 10);
            *iptr = 30;
            assert_eq!(*iptr, 30);
            assert_eq!(p.fst, 30);

            *pptr = Pair {fst: 50, snd: 60};
            assert_eq!(*iptr, 50);
            assert_eq!(p.fst, 50);
            assert_eq!(p.snd, 60);

            let v0 = ~[32000u16, 32001u16, 32002u16];
            let mut v1 = ~[0u16, 0u16, 0u16];

            copy_memory(mut_offset(v1.as_mut_ptr(), 1),
                        offset(v0.as_ptr(), 1), 1);
            assert!((v1[0] == 0u16 && v1[1] == 32001u16 && v1[2] == 0u16));
            copy_memory(v1.as_mut_ptr(),
                        offset(v0.as_ptr(), 2), 1);
            assert!((v1[0] == 32002u16 && v1[1] == 32001u16 &&
                     v1[2] == 0u16));
            copy_memory(mut_offset(v1.as_mut_ptr(), 2),
                        v0.as_ptr(), 1u);
            assert!((v1[0] == 32002u16 && v1[1] == 32001u16 &&
                     v1[2] == 32000u16));
        }
    }

    #[test]
    fn test_position() {
        "hello".with_c_str(|p| {
            unsafe {
                assert!(2u == position(p, |c| *c == 'l' as libc::c_char));
                assert!(4u == position(p, |c| *c == 'o' as libc::c_char));
                assert!(5u == position(p, |c| *c == 0 as libc::c_char));
            }
        })
    }

    #[test]
    fn test_buf_len() {
        "hello".with_c_str(|p0| {
            "there".with_c_str(|p1| {
                "thing".with_c_str(|p2| {
                    let v = ~[p0, p1, p2, null()];
                    unsafe {
                        assert_eq!(buf_len(v.as_ptr()), 3u);
                    }
                })
            })
        })
    }

    #[test]
    fn test_is_null() {
        let p: *int = null();
        assert!(p.is_null());
        assert!(!p.is_not_null());

        let q = unsafe { offset(p, 1) };
        assert!(!q.is_null());
        assert!(q.is_not_null());

        let mp: *mut int = mut_null();
        assert!(mp.is_null());
        assert!(!mp.is_not_null());

        let mq = unsafe { mp.offset(1) };
        assert!(!mq.is_null());
        assert!(mq.is_not_null());
    }

    #[test]
    fn test_ptr_addition() {
        unsafe {
            let xs = ~[5, ..16];
            let mut ptr = xs.as_ptr();
            let end = ptr.offset(16);

            while ptr < end {
                assert_eq!(*ptr, 5);
                ptr = ptr.offset(1);
            }

            let mut xs_mut = xs.clone();
            let mut m_ptr = xs_mut.as_mut_ptr();
            let m_end = m_ptr.offset(16);

            while m_ptr < m_end {
                *m_ptr += 5;
                m_ptr = m_ptr.offset(1);
            }

            assert_eq!(xs_mut, ~[10, ..16]);
        }
    }

    #[test]
    fn test_ptr_subtraction() {
        unsafe {
            let xs = ~[0,1,2,3,4,5,6,7,8,9];
            let mut idx = 9i8;
            let ptr = xs.as_ptr();

            while idx >= 0i8 {
                assert_eq!(*(ptr.offset(idx as int)), idx as int);
                idx = idx - 1i8;
            }

            let mut xs_mut = xs.clone();
            let m_start = xs_mut.as_mut_ptr();
            let mut m_ptr = m_start.offset(9);

            while m_ptr >= m_start {
                *m_ptr += *m_ptr;
                m_ptr = m_ptr.offset(-1);
            }

            assert_eq!(xs_mut, ~[0,2,4,6,8,10,12,14,16,18]);
        }
    }

    #[test]
    fn test_set_memory() {
        let mut xs = [0u8, ..20];
        let ptr = xs.as_mut_ptr();
        unsafe { set_memory(ptr, 5u8, xs.len()); }
        assert_eq!(xs, [5u8, ..20]);
    }
}
