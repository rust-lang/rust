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

use cast;
use clone::Clone;
use iterator::{range, Iterator};
use option::{Option, Some, None};
use unstable::intrinsics;
use util::swap;

#[cfg(not(test))] use ops::{Add,Sub};
#[cfg(not(test))] use num::Int;

#[cfg(not(test))] use cmp::{Eq, Ord};

/// Calculate the offset from a pointer
#[inline]
pub fn offset<T>(ptr: *T, count: int) -> *T {
    unsafe { intrinsics::offset(ptr, count) }
}

/// Calculate the offset from a const pointer
#[inline]
pub fn const_offset<T>(ptr: *const T, count: int) -> *const T {
    unsafe { intrinsics::offset(ptr as *T, count) }
}

/// Calculate the offset from a mut pointer
#[inline]
pub fn mut_offset<T>(ptr: *mut T, count: int) -> *mut T {
    unsafe { intrinsics::offset(ptr as *T, count) as *mut T }
}

/// Return the offset of the first null pointer in `buf`.
#[inline]
pub unsafe fn buf_len<T>(buf: **T) -> uint {
    position(buf, |i| *i == null())
}

impl<T> Clone for *T {
    #[inline]
    fn clone(&self) -> *T {
        *self
    }
}

/// Return the first offset `i` such that `f(buf[i]) == true`.
#[inline]
pub unsafe fn position<T>(buf: *T, f: &fn(&T) -> bool) -> uint {
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
pub fn is_null<T>(ptr: *const T) -> bool { ptr == null() }

/// Returns true if the pointer is not equal to the null pointer.
#[inline]
pub fn is_not_null<T>(ptr: *const T) -> bool { !is_null(ptr) }

/**
 * Copies data from one location to another.
 *
 * Copies `count` elements (not bytes) from `src` to `dst`. The source
 * and destination may overlap.
 */
#[inline]
#[cfg(target_word_size = "32")]
pub unsafe fn copy_memory<T>(dst: *mut T, src: *const T, count: uint) {
    intrinsics::memmove32(dst, src as *T, count as u32);
}

/**
 * Copies data from one location to another.
 *
 * Copies `count` elements (not bytes) from `src` to `dst`. The source
 * and destination may overlap.
 */
#[inline]
#[cfg(target_word_size = "64")]
pub unsafe fn copy_memory<T>(dst: *mut T, src: *const T, count: uint) {
    intrinsics::memmove64(dst, src as *T, count as u64);
}

/**
 * Copies data from one location to another.
 *
 * Copies `count` elements (not bytes) from `src` to `dst`. The source
 * and destination may *not* overlap.
 */
#[inline]
#[cfg(target_word_size = "32")]
pub unsafe fn copy_nonoverlapping_memory<T>(dst: *mut T, src: *const T, count: uint) {
    intrinsics::memcpy32(dst, src as *T, count as u32);
}

/**
 * Copies data from one location to another.
 *
 * Copies `count` elements (not bytes) from `src` to `dst`. The source
 * and destination may *not* overlap.
 */
#[inline]
#[cfg(target_word_size = "64")]
pub unsafe fn copy_nonoverlapping_memory<T>(dst: *mut T, src: *const T, count: uint) {
    intrinsics::memcpy64(dst, src as *T, count as u64);
}

/**
 * Invokes memset on the specified pointer, setting `count * size_of::<T>()`
 * bytes of memory starting at `dst` to `c`.
 */
#[inline]
#[cfg(target_word_size = "32")]
pub unsafe fn set_memory<T>(dst: *mut T, c: u8, count: uint) {
    intrinsics::memset32(dst, c, count as u32);
}

/**
 * Invokes memset on the specified pointer, setting `count * size_of::<T>()`
 * bytes of memory starting at `dst` to `c`.
 */
#[inline]
#[cfg(target_word_size = "64")]
pub unsafe fn set_memory<T>(dst: *mut T, c: u8, count: uint) {
    intrinsics::memset64(dst, c, count as u64);
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
    copy_nonoverlapping_memory(t, x, 1);
    copy_memory(x, y, 1); // `x` and `y` may overlap
    copy_nonoverlapping_memory(y, t, 1);

    // y and t now point to the same thing, but we need to completely forget `tmp`
    // because it's no longer relevant.
    cast::forget(tmp);
}

/**
 * Replace the value at a mutable location with a new one, returning the old
 * value, without deinitialising or copying either one.
 */
#[inline]
pub unsafe fn replace_ptr<T>(dest: *mut T, mut src: T) -> T {
    swap(cast::transmute(dest), &mut src); // cannot overlap
    src
}

/**
 * Reads the value from `*src` and returns it. Does not copy `*src`.
 */
#[inline(always)]
pub unsafe fn read_ptr<T>(src: *mut T) -> T {
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
    let tmp = read_ptr(dest);

    // Now zero out `dest`:
    zero_memory(dest, 1);

    tmp
}

/// Transform a region pointer - &T - to an unsafe pointer - *T.
#[inline]
pub fn to_unsafe_ptr<T>(thing: &T) -> *T {
    thing as *T
}

/// Transform a const region pointer - &const T - to a const unsafe pointer - *const T.
#[inline]
pub fn to_const_unsafe_ptr<T>(thing: &const T) -> *const T {
    thing as *const T
}

/// Transform a mutable region pointer - &mut T - to a mutable unsafe pointer - *mut T.
#[inline]
pub fn to_mut_unsafe_ptr<T>(thing: &mut T) -> *mut T {
    thing as *mut T
}

/**
  Given a **T (pointer to an array of pointers),
  iterate through each *T, up to the provided `len`,
  passing to the provided callback function

  SAFETY NOTE: Pointer-arithmetic. Dragons be here.
*/
pub unsafe fn array_each_with_len<T>(arr: **T, len: uint, cb: &fn(*T)) {
    debug!("array_each_with_len: before iterate");
    if (arr as uint == 0) {
        fail!("ptr::array_each_with_len failure: arr input is null pointer");
    }
    //let start_ptr = *arr;
    for e in range(0, len) {
        let n = offset(arr, e as int);
        cb(*n);
    }
    debug!("array_each_with_len: after iterate");
}

/**
  Given a null-pointer-terminated **T (pointer to
  an array of pointers), iterate through each *T,
  passing to the provided callback function

  SAFETY NOTE: This will only work with a null-terminated
  pointer array. Barely less-dodgy Pointer Arithmetic.
  Dragons be here.
*/
pub unsafe fn array_each<T>(arr: **T, cb: &fn(*T)) {
    if (arr as uint == 0) {
        fail!("ptr::array_each_with_len failure: arr input is null pointer");
    }
    let len = buf_len(arr);
    debug!("array_each inferred len: %u",
                    len);
    array_each_with_len(arr, len, cb);
}

#[allow(missing_doc)]
pub trait RawPtr<T> {
    fn is_null(&self) -> bool;
    fn is_not_null(&self) -> bool;
    unsafe fn to_option(&self) -> Option<&T>;
    fn offset(&self, count: int) -> Self;
    unsafe fn offset_inbounds(self, count: int) -> Self;
}

/// Extension methods for immutable pointers
impl<T> RawPtr<T> for *T {
    /// Returns true if the pointer is equal to the null pointer.
    #[inline]
    fn is_null(&self) -> bool { is_null(*self) }

    /// Returns true if the pointer is not equal to the null pointer.
    #[inline]
    fn is_not_null(&self) -> bool { is_not_null(*self) }

    ///
    /// Returns `None` if the pointer is null, or else returns the value wrapped
    /// in `Some`.
    ///
    /// # Safety Notes
    ///
    /// While this method is useful for null-safety, it is important to note
    /// that this is still an unsafe operation because the returned value could
    /// be pointing to invalid memory.
    ///
    #[inline]
    unsafe fn to_option(&self) -> Option<&T> {
        if self.is_null() { None } else {
            Some(cast::transmute(*self))
        }
    }

    /// Calculates the offset from a pointer.
    #[inline]
    fn offset(&self, count: int) -> *T { offset(*self, count) }

    /// Calculates the offset from a pointer. The offset *must* be in-bounds of
    /// the object, or one-byte-past-the-end.
    #[inline]
    unsafe fn offset_inbounds(self, count: int) -> *T {
        intrinsics::offset_inbounds(self, count)
    }
}

/// Extension methods for mutable pointers
impl<T> RawPtr<T> for *mut T {
    /// Returns true if the pointer is equal to the null pointer.
    #[inline]
    fn is_null(&self) -> bool { is_null(*self) }

    /// Returns true if the pointer is not equal to the null pointer.
    #[inline]
    fn is_not_null(&self) -> bool { is_not_null(*self) }

    ///
    /// Returns `None` if the pointer is null, or else returns the value wrapped
    /// in `Some`.
    ///
    /// # Safety Notes
    ///
    /// While this method is useful for null-safety, it is important to note
    /// that this is still an unsafe operation because the returned value could
    /// be pointing to invalid memory.
    ///
    #[inline]
    unsafe fn to_option(&self) -> Option<&T> {
        if self.is_null() { None } else {
            Some(cast::transmute(*self))
        }
    }

    /// Calculates the offset from a mutable pointer.
    #[inline]
    fn offset(&self, count: int) -> *mut T { mut_offset(*self, count) }

    /// Calculates the offset from a pointer. The offset *must* be in-bounds of
    /// the object, or one-byte-past-the-end. An arithmetic overflow is also
    /// undefined behaviour.
    ///
    /// This method should be preferred over `offset` when the guarantee can be
    /// satisfied, to enable better optimization.
    #[inline]
    unsafe fn offset_inbounds(self, count: int) -> *mut T {
        intrinsics::offset_inbounds(self as *T, count) as *mut T
    }
}

// Equality for pointers
#[cfg(not(test))]
impl<T> Eq for *const T {
    #[inline]
    fn eq(&self, other: &*const T) -> bool {
        (*self as uint) == (*other as uint)
    }
    #[inline]
    fn ne(&self, other: &*const T) -> bool { !self.eq(other) }
}

// Equality for extern "C" fn pointers
#[cfg(not(test))]
mod externfnpointers {
    use cast;
    use cmp::Eq;

    impl<_R> Eq for extern "C" fn() -> _R {
        #[inline]
        fn eq(&self, other: &extern "C" fn() -> _R) -> bool {
            let self_: *() = unsafe { cast::transmute(*self) };
            let other_: *() = unsafe { cast::transmute(*other) };
            self_ == other_
        }
        #[inline]
        fn ne(&self, other: &extern "C" fn() -> _R) -> bool {
            !self.eq(other)
        }
    }
    macro_rules! fnptreq(
        ($($p:ident),*) => {
            impl<_R,$($p),*> Eq for extern "C" fn($($p),*) -> _R {
                #[inline]
                fn eq(&self, other: &extern "C" fn($($p),*) -> _R) -> bool {
                    let self_: *() = unsafe { cast::transmute(*self) };
                    let other_: *() = unsafe { cast::transmute(*other) };
                    self_ == other_
                }
                #[inline]
                fn ne(&self, other: &extern "C" fn($($p),*) -> _R) -> bool {
                    !self.eq(other)
                }
            }
        }
    )
    fnptreq!(A)
    fnptreq!(A,B)
    fnptreq!(A,B,C)
    fnptreq!(A,B,C,D)
    fnptreq!(A,B,C,D,E)
}

// Comparison for pointers
#[cfg(not(test))]
impl<T> Ord for *const T {
    #[inline]
    fn lt(&self, other: &*const T) -> bool {
        (*self as uint) < (*other as uint)
    }
    #[inline]
    fn le(&self, other: &*const T) -> bool {
        (*self as uint) <= (*other as uint)
    }
    #[inline]
    fn ge(&self, other: &*const T) -> bool {
        (*self as uint) >= (*other as uint)
    }
    #[inline]
    fn gt(&self, other: &*const T) -> bool {
        (*self as uint) > (*other as uint)
    }
}

#[cfg(not(test))]
impl<T, I: Int> Add<I, *T> for *T {
    /// Add an integer value to a pointer to get an offset pointer.
    /// Is calculated according to the size of the type pointed to.
    #[inline]
    fn add(&self, rhs: &I) -> *T {
        self.offset(rhs.to_int() as int)
    }
}

#[cfg(not(test))]
impl<T, I: Int> Sub<I, *T> for *T {
    /// Subtract an integer value from a pointer to get an offset pointer.
    /// Is calculated according to the size of the type pointed to.
    #[inline]
    fn sub(&self, rhs: &I) -> *T {
        self.offset(-rhs.to_int() as int)
    }
}

#[cfg(not(test))]
impl<T, I: Int> Add<I, *mut T> for *mut T {
    /// Add an integer value to a pointer to get an offset pointer.
    /// Is calculated according to the size of the type pointed to.
    #[inline]
    fn add(&self, rhs: &I) -> *mut T {
        self.offset(rhs.to_int() as int)
    }
}

#[cfg(not(test))]
impl<T, I: Int> Sub<I, *mut T> for *mut T {
    /// Subtract an integer value from a pointer to get an offset pointer.
    /// Is calculated according to the size of the type pointed to.
    #[inline]
    fn sub(&self, rhs: &I) -> *mut T {
        self.offset(-rhs.to_int() as int)
    }
}

#[cfg(test)]
pub mod ptr_tests {
    use super::*;
    use prelude::*;

    use c_str::ToCStr;
    use cast;
    use libc;
    use str;
    use vec;

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

            copy_memory(mut_offset(vec::raw::to_mut_ptr(v1), 1),
                        offset(vec::raw::to_ptr(v0), 1), 1);
            assert!((v1[0] == 0u16 && v1[1] == 32001u16 && v1[2] == 0u16));
            copy_memory(vec::raw::to_mut_ptr(v1),
                        offset(vec::raw::to_ptr(v0), 2), 1);
            assert!((v1[0] == 32002u16 && v1[1] == 32001u16 &&
                     v1[2] == 0u16));
            copy_memory(mut_offset(vec::raw::to_mut_ptr(v1), 2),
                        vec::raw::to_ptr(v0), 1u);
            assert!((v1[0] == 32002u16 && v1[1] == 32001u16 &&
                     v1[2] == 32000u16));
        }
    }

    #[test]
    fn test_position() {
        use libc::c_char;

        do "hello".with_c_str |p| {
            unsafe {
                assert!(2u == position(p, |c| *c == 'l' as c_char));
                assert!(4u == position(p, |c| *c == 'o' as c_char));
                assert!(5u == position(p, |c| *c == 0 as c_char));
            }
        }
    }

    #[test]
    fn test_buf_len() {
        do "hello".with_c_str |p0| {
            do "there".with_c_str |p1| {
                do "thing".with_c_str |p2| {
                    let v = ~[p0, p1, p2, null()];
                    do v.as_imm_buf |vp, len| {
                        assert_eq!(unsafe { buf_len(vp) }, 3u);
                        assert_eq!(len, 4u);
                    }
                }
            }
        }
    }

    #[test]
    fn test_is_null() {
        let p: *int = null();
        assert!(p.is_null());
        assert!(!p.is_not_null());

        let q = offset(p, 1);
        assert!(!q.is_null());
        assert!(q.is_not_null());

        let mp: *mut int = mut_null();
        assert!(mp.is_null());
        assert!(!mp.is_not_null());

        let mq = mp.offset(1);
        assert!(!mq.is_null());
        assert!(mq.is_not_null());
    }

    #[test]
    fn test_to_option() {
        unsafe {
            let p: *int = null();
            assert_eq!(p.to_option(), None);

            let q: *int = &2;
            assert_eq!(q.to_option().unwrap(), &2);

            let p: *mut int = mut_null();
            assert_eq!(p.to_option(), None);

            let q: *mut int = &mut 2;
            assert_eq!(q.to_option().unwrap(), &2);
        }
    }

    #[test]
    fn test_ptr_addition() {
        use vec::raw::*;

        unsafe {
            let xs = ~[5, ..16];
            let mut ptr = to_ptr(xs);
            let end = ptr + 16;

            while ptr < end {
                assert_eq!(*ptr, 5);
                ptr = ptr + 1u;
            }

            let mut xs_mut = xs.clone();
            let mut m_ptr = to_mut_ptr(xs_mut);
            let m_end = m_ptr + 16i16;

            while m_ptr < m_end {
                *m_ptr += 5;
                m_ptr = m_ptr + 1u8;
            }

            assert_eq!(xs_mut, ~[10, ..16]);
        }
    }

    #[test]
    fn test_ptr_subtraction() {
        use vec::raw::*;

        unsafe {
            let xs = ~[0,1,2,3,4,5,6,7,8,9];
            let mut idx = 9i8;
            let ptr = to_ptr(xs);

            while idx >= 0i8 {
                assert_eq!(*(ptr + idx), idx as int);
                idx = idx - 1i8;
            }

            let mut xs_mut = xs.clone();
            let m_start = to_mut_ptr(xs_mut);
            let mut m_ptr = m_start + 9u32;

            while m_ptr >= m_start {
                *m_ptr += *m_ptr;
                m_ptr = m_ptr - 1i8;
            }

            assert_eq!(xs_mut, ~[0,2,4,6,8,10,12,14,16,18]);
        }
    }

    #[test]
    fn test_ptr_array_each_with_len() {
        unsafe {
            let one = "oneOne".to_c_str();
            let two = "twoTwo".to_c_str();
            let three = "threeThree".to_c_str();
            let arr = ~[
                one.with_ref(|buf| buf),
                two.with_ref(|buf| buf),
                three.with_ref(|buf| buf),
            ];
            let expected_arr = [
                one, two, three
            ];

            do arr.as_imm_buf |arr_ptr, arr_len| {
                let mut ctr = 0;
                let mut iteration_count = 0;
                do array_each_with_len(arr_ptr, arr_len) |e| {
                     let actual = str::raw::from_c_str(e);
                     let expected = do expected_arr[ctr].with_ref |buf| {
                         str::raw::from_c_str(buf)
                     };
                     debug!(
                         "test_ptr_array_each_with_len e: %s, a: %s",
                         expected, actual);
                     assert_eq!(actual, expected);
                     ctr += 1;
                     iteration_count += 1;
                 }
                assert_eq!(iteration_count, 3u);
            }
        }
    }

    #[test]
    fn test_ptr_array_each() {
        unsafe {
            let one = "oneOne".to_c_str();
            let two = "twoTwo".to_c_str();
            let three = "threeThree".to_c_str();
            let arr = ~[
                one.with_ref(|buf| buf),
                two.with_ref(|buf| buf),
                three.with_ref(|buf| buf),
                // fake a null terminator
                null(),
            ];
            let expected_arr = [
                one, two, three
            ];

            do arr.as_imm_buf |arr_ptr, _| {
                let mut ctr = 0;
                let mut iteration_count = 0;
                do array_each(arr_ptr) |e| {
                     let actual = str::raw::from_c_str(e);
                     let expected = do expected_arr[ctr].with_ref |buf| {
                         str::raw::from_c_str(buf)
                     };
                     debug!(
                         "test_ptr_array_each e: %s, a: %s",
                         expected, actual);
                     assert_eq!(actual, expected);
                     ctr += 1;
                     iteration_count += 1;
                 }
                assert_eq!(iteration_count, 3);
            }
        }
    }

    #[test]
    #[should_fail]
    fn test_ptr_array_each_with_len_null_ptr() {
        unsafe {
            array_each_with_len(0 as **libc::c_char, 1, |e| {
                str::raw::from_c_str(e);
            });
        }
    }
    #[test]
    #[should_fail]
    fn test_ptr_array_each_null_ptr() {
        unsafe {
            array_each(0 as **libc::c_char, |e| {
                str::raw::from_c_str(e);
            });
        }
    }

    #[test]
    fn test_set_memory() {
        let mut xs = [0u8, ..20];
        let ptr = vec::raw::to_mut_ptr(xs);
        unsafe { set_memory(ptr, 5u8, xs.len()); }
        assert_eq!(xs, [5u8, ..20]);
    }
}
