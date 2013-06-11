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
use option::{Option, Some, None};
use sys;
use unstable::intrinsics;

#[cfg(not(test))] use cmp::{Eq, Ord};
use uint;

/// Calculate the offset from a pointer
#[inline(always)]
pub fn offset<T>(ptr: *T, count: uint) -> *T {
    (ptr as uint + count * sys::size_of::<T>()) as *T
}

/// Calculate the offset from a const pointer
#[inline(always)]
pub fn const_offset<T>(ptr: *const T, count: uint) -> *const T {
    (ptr as uint + count * sys::size_of::<T>()) as *T
}

/// Calculate the offset from a mut pointer
#[inline(always)]
pub fn mut_offset<T>(ptr: *mut T, count: uint) -> *mut T {
    (ptr as uint + count * sys::size_of::<T>()) as *mut T
}

/// Return the offset of the first null pointer in `buf`.
#[inline(always)]
pub unsafe fn buf_len<T>(buf: **T) -> uint {
    position(buf, |i| *i == null())
}

/// Return the first offset `i` such that `f(buf[i]) == true`.
#[inline(always)]
pub unsafe fn position<T>(buf: *T, f: &fn(&T) -> bool) -> uint {
    let mut i = 0;
    loop {
        if f(&(*offset(buf, i))) { return i; }
        else { i += 1; }
    }
}

/// Create an unsafe null pointer
#[inline(always)]
pub fn null<T>() -> *T { 0 as *T }

/// Create an unsafe mutable null pointer
#[inline(always)]
pub fn mut_null<T>() -> *mut T { 0 as *mut T }

/// Returns true if the pointer is equal to the null pointer.
#[inline(always)]
pub fn is_null<T>(ptr: *const T) -> bool { ptr == null() }

/// Returns true if the pointer is not equal to the null pointer.
#[inline(always)]
pub fn is_not_null<T>(ptr: *const T) -> bool { !is_null(ptr) }

/**
 * Copies data from one location to another.
 *
 * Copies `count` elements (not bytes) from `src` to `dst`. The source
 * and destination may overlap.
 */
#[inline(always)]
#[cfg(target_word_size = "32", stage0)]
pub unsafe fn copy_memory<T>(dst: *mut T, src: *const T, count: uint) {
    use unstable::intrinsics::memmove32;
    let n = count * sys::size_of::<T>();
    memmove32(dst as *mut u8, src as *u8, n as u32);
}

/**
 * Copies data from one location to another.
 *
 * Copies `count` elements (not bytes) from `src` to `dst`. The source
 * and destination may overlap.
 */
#[inline(always)]
#[cfg(target_word_size = "32", not(stage0))]
pub unsafe fn copy_memory<T>(dst: *mut T, src: *const T, count: uint) {
    use unstable::intrinsics::memmove32;
    memmove32(dst, src as *T, count as u32);
}

/**
 * Copies data from one location to another.
 *
 * Copies `count` elements (not bytes) from `src` to `dst`. The source
 * and destination may overlap.
 */
#[inline(always)]
#[cfg(target_word_size = "64", stage0)]
pub unsafe fn copy_memory<T>(dst: *mut T, src: *const T, count: uint) {
    use unstable::intrinsics::memmove64;
    let n = count * sys::size_of::<T>();
    memmove64(dst as *mut u8, src as *u8, n as u64);
}

/**
 * Copies data from one location to another.
 *
 * Copies `count` elements (not bytes) from `src` to `dst`. The source
 * and destination may overlap.
 */
#[inline(always)]
#[cfg(target_word_size = "64", not(stage0))]
pub unsafe fn copy_memory<T>(dst: *mut T, src: *const T, count: uint) {
    use unstable::intrinsics::memmove64;
    memmove64(dst, src as *T, count as u64);
}

/**
 * Copies data from one location to another.
 *
 * Copies `count` elements (not bytes) from `src` to `dst`. The source
 * and destination may *not* overlap.
 */
#[inline(always)]
#[cfg(target_word_size = "32", stage0)]
pub unsafe fn copy_nonoverlapping_memory<T>(dst: *mut T, src: *const T, count: uint) {
    use unstable::intrinsics::memmove32;
    let n = count * sys::size_of::<T>();
    memmove32(dst as *mut u8, src as *u8, n as u32);
}

/**
 * Copies data from one location to another.
 *
 * Copies `count` elements (not bytes) from `src` to `dst`. The source
 * and destination may *not* overlap.
 */
#[inline(always)]
#[cfg(target_word_size = "32", not(stage0))]
pub unsafe fn copy_nonoverlapping_memory<T>(dst: *mut T, src: *const T, count: uint) {
    use unstable::intrinsics::memcpy32;
    memcpy32(dst, src as *T, count as u32);
}

/**
 * Copies data from one location to another.
 *
 * Copies `count` elements (not bytes) from `src` to `dst`. The source
 * and destination may *not* overlap.
 */
#[inline(always)]
#[cfg(target_word_size = "64", stage0)]
pub unsafe fn copy_nonoverlapping_memory<T>(dst: *mut T, src: *const T, count: uint) {
    use unstable::intrinsics::memmove64;
    let n = count * sys::size_of::<T>();
    memmove64(dst as *mut u8, src as *u8, n as u64);
}

/**
 * Copies data from one location to another.
 *
 * Copies `count` elements (not bytes) from `src` to `dst`. The source
 * and destination may *not* overlap.
 */
#[inline(always)]
#[cfg(target_word_size = "64", not(stage0))]
pub unsafe fn copy_nonoverlapping_memory<T>(dst: *mut T, src: *const T, count: uint) {
    use unstable::intrinsics::memcpy64;
    memcpy64(dst, src as *T, count as u64);
}

/**
 * Invokes memset on the specified pointer, setting `count * size_of::<T>()`
 * bytes of memory starting at `dst` to `c`.
 */
#[inline(always)]
#[cfg(target_word_size = "32", not(stage0))]
pub unsafe fn set_memory<T>(dst: *mut T, c: u8, count: uint) {
    use unstable::intrinsics::memset32;
    memset32(dst, c, count as u32);
}

/**
 * Invokes memset on the specified pointer, setting `count * size_of::<T>()`
 * bytes of memory starting at `dst` to `c`.
 */
#[inline(always)]
#[cfg(target_word_size = "64", not(stage0))]
pub unsafe fn set_memory<T>(dst: *mut T, c: u8, count: uint) {
    use unstable::intrinsics::memset64;
    memset64(dst, c, count as u64);
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
    copy_memory(t, x, 1);
    copy_memory(x, y, 1);
    copy_memory(y, t, 1);

    // y and t now point to the same thing, but we need to completely forget `tmp`
    // because it's no longer relevant.
    cast::forget(tmp);
}

/**
 * Replace the value at a mutable location with a new one, returning the old
 * value, without deinitialising or copying either one.
 */
#[inline(always)]
pub unsafe fn replace_ptr<T>(dest: *mut T, mut src: T) -> T {
    swap_ptr(dest, &mut src);
    src
}

/// Transform a region pointer - &T - to an unsafe pointer - *T.
#[inline(always)]
pub fn to_unsafe_ptr<T>(thing: &T) -> *T {
    thing as *T
}

/// Transform a const region pointer - &const T - to a const unsafe pointer - *const T.
#[inline(always)]
pub fn to_const_unsafe_ptr<T>(thing: &const T) -> *const T {
    thing as *const T
}

/// Transform a mutable region pointer - &mut T - to a mutable unsafe pointer - *mut T.
#[inline(always)]
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
    uint::iterate(0, len, |e| {
        let n = offset(arr, e);
        cb(*n);
        true
    });
    debug!("array_each_with_len: after iterate");
}

/**
  Given a null-pointer-terminated **T (pointer to
  an array of pointers), iterate through each *T,
  passing to the provided callback function

  SAFETY NOTE: This will only work with a null-terminated
  pointer array. Barely less-dodgey Pointer Arithmetic.
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
    fn is_null(&const self) -> bool;
    fn is_not_null(&const self) -> bool;
    unsafe fn to_option(&const self) -> Option<&T>;
    fn offset(&self, count: uint) -> Self;
}

/// Extension methods for immutable pointers
impl<T> RawPtr<T> for *T {
    /// Returns true if the pointer is equal to the null pointer.
    #[inline(always)]
    fn is_null(&const self) -> bool { is_null(*self) }

    /// Returns true if the pointer is not equal to the null pointer.
    #[inline(always)]
    fn is_not_null(&const self) -> bool { is_not_null(*self) }

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
    #[inline(always)]
    unsafe fn to_option(&const self) -> Option<&T> {
        if self.is_null() { None } else {
            Some(cast::transmute(*self))
        }
    }

    /// Calculates the offset from a pointer.
    #[inline(always)]
    fn offset(&self, count: uint) -> *T { offset(*self, count) }
}

/// Extension methods for mutable pointers
impl<T> RawPtr<T> for *mut T {
    /// Returns true if the pointer is equal to the null pointer.
    #[inline(always)]
    fn is_null(&const self) -> bool { is_null(*self) }

    /// Returns true if the pointer is not equal to the null pointer.
    #[inline(always)]
    fn is_not_null(&const self) -> bool { is_not_null(*self) }

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
    #[inline(always)]
    unsafe fn to_option(&const self) -> Option<&T> {
        if self.is_null() { None } else {
            Some(cast::transmute(*self))
        }
    }

    /// Calculates the offset from a mutable pointer.
    #[inline(always)]
    fn offset(&self, count: uint) -> *mut T { mut_offset(*self, count) }
}

// Equality for pointers
#[cfg(not(test))]
impl<T> Eq for *const T {
    #[inline(always)]
    fn eq(&self, other: &*const T) -> bool {
        (*self as uint) == (*other as uint)
    }
    #[inline(always)]
    fn ne(&self, other: &*const T) -> bool { !self.eq(other) }
}

// Comparison for pointers
#[cfg(not(test))]
impl<T> Ord for *const T {
    #[inline(always)]
    fn lt(&self, other: &*const T) -> bool {
        (*self as uint) < (*other as uint)
    }
    #[inline(always)]
    fn le(&self, other: &*const T) -> bool {
        (*self as uint) <= (*other as uint)
    }
    #[inline(always)]
    fn ge(&self, other: &*const T) -> bool {
        (*self as uint) >= (*other as uint)
    }
    #[inline(always)]
    fn gt(&self, other: &*const T) -> bool {
        (*self as uint) > (*other as uint)
    }
}

#[cfg(test)]
pub mod ptr_tests {
    use super::*;
    use prelude::*;

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

            copy_memory(mut_offset(vec::raw::to_mut_ptr(v1), 1u),
                        offset(vec::raw::to_ptr(v0), 1u), 1u);
            assert!((v1[0] == 0u16 && v1[1] == 32001u16 && v1[2] == 0u16));
            copy_memory(vec::raw::to_mut_ptr(v1),
                        offset(vec::raw::to_ptr(v0), 2u), 1u);
            assert!((v1[0] == 32002u16 && v1[1] == 32001u16 &&
                     v1[2] == 0u16));
            copy_memory(mut_offset(vec::raw::to_mut_ptr(v1), 2u),
                        vec::raw::to_ptr(v0), 1u);
            assert!((v1[0] == 32002u16 && v1[1] == 32001u16 &&
                     v1[2] == 32000u16));
        }
    }

    #[test]
    fn test_position() {
        use str::as_c_str;
        use libc::c_char;

        let s = ~"hello";
        unsafe {
            assert!(2u == as_c_str(s, |p| position(p,
                                                   |c| *c == 'l' as c_char)));
            assert!(4u == as_c_str(s, |p| position(p,
                                                   |c| *c == 'o' as c_char)));
            assert!(5u == as_c_str(s, |p| position(p,
                                                   |c| *c == 0 as c_char)));
        }
    }

    #[test]
    fn test_buf_len() {
        let s0 = ~"hello";
        let s1 = ~"there";
        let s2 = ~"thing";
        do str::as_c_str(s0) |p0| {
            do str::as_c_str(s1) |p1| {
                do str::as_c_str(s2) |p2| {
                    let v = ~[p0, p1, p2, null()];
                    do vec::as_imm_buf(v) |vp, len| {
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

        let q = offset(p, 1u);
        assert!(!q.is_null());
        assert!(q.is_not_null());

        let mp: *mut int = mut_null();
        assert!(mp.is_null());
        assert!(!mp.is_not_null());

        let mq = mp.offset(1u);
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
    fn test_ptr_array_each_with_len() {
        unsafe {
            let one = ~"oneOne";
            let two = ~"twoTwo";
            let three = ~"threeThree";
            let arr: ~[*i8] = ~[
                ::cast::transmute(&one[0]),
                ::cast::transmute(&two[0]),
                ::cast::transmute(&three[0]),
            ];
            let expected_arr = [
                one, two, three
            ];
            let arr_ptr = &arr[0];
            let mut ctr = 0;
            let mut iteration_count = 0;
            array_each_with_len(arr_ptr, arr.len(),
                                |e| {
                                         let actual = str::raw::from_c_str(e);
                                         let expected = copy expected_arr[ctr];
                                         debug!(
                                             "test_ptr_array_each e: %s, a: %s",
                                             expected, actual);
                                         assert_eq!(actual, expected);
                                         ctr += 1;
                                         iteration_count += 1;
                                     });
            assert_eq!(iteration_count, 3u);
        }
    }
    #[test]
    fn test_ptr_array_each() {
        unsafe {
            let one = ~"oneOne";
            let two = ~"twoTwo";
            let three = ~"threeThree";
            let arr: ~[*i8] = ~[
                ::cast::transmute(&one[0]),
                ::cast::transmute(&two[0]),
                ::cast::transmute(&three[0]),
                // fake a null terminator
                0 as *i8
            ];
            let expected_arr = [
                one, two, three
            ];
            let arr_ptr = &arr[0];
            let mut ctr = 0;
            let mut iteration_count = 0;
            array_each(arr_ptr, |e| {
                let actual = str::raw::from_c_str(e);
                let expected = copy expected_arr[ctr];
                debug!(
                    "test_ptr_array_each e: %s, a: %s",
                    expected, actual);
                assert_eq!(actual, expected);
                ctr += 1;
                iteration_count += 1;
            });
            assert_eq!(iteration_count, 3);
        }
    }
    #[test]
    #[should_fail]
    #[ignore(cfg(windows))]
    fn test_ptr_array_each_with_len_null_ptr() {
        unsafe {
            array_each_with_len(0 as **libc::c_char, 1, |e| {
                str::raw::from_c_str(e);
            });
        }
    }
    #[test]
    #[should_fail]
    #[ignore(cfg(windows))]
    fn test_ptr_array_each_null_ptr() {
        unsafe {
            array_each(0 as **libc::c_char, |e| {
                str::raw::from_c_str(e);
            });
        }
    }

    #[test]
    #[cfg(not(stage0))]
    fn test_set_memory() {
        let mut xs = [0u8, ..20];
        let ptr = vec::raw::to_mut_ptr(xs);
        unsafe { set_memory(ptr, 5u8, xs.len()); }
        assert_eq!(xs, [5u8, ..20]);
    }
}
