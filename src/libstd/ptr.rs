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
#[cfg(not(test))]
use cmp::Equiv;
use iter::{range, Iterator};
use mem;
use option::{Option, Some, None};
use unstable::intrinsics;

#[cfg(not(test))] use cmp::{Eq, Ord};

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

impl<T> Clone for *mut T {
    #[inline]
    fn clone(&self) -> *mut T {
        *self
    }
}

/// Return the first offset `i` such that `f(buf[i]) == true`.
#[inline]
pub unsafe fn position<T>(buf: *T, f: |&T| -> bool) -> uint {
    let mut i = 0;
    loop {
        if f(&(*buf.offset(i as int))) { return i; }
        else { i += 1; }
    }
}

/// Create an unsafe null pointer
#[inline]
pub fn null<T>() -> *T { 0 as *T }

/// Create an unsafe mutable null pointer
#[inline]
pub fn mut_null<T>() -> *mut T { 0 as *mut T }

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
 * deinitialising either. They may overlap.
 */
#[inline]
pub unsafe fn swap<T>(x: *mut T, y: *mut T) {
    // Give ourselves some scratch space to work with
    let mut tmp: T = mem::uninit();
    let t: *mut T = &mut tmp;

    // Perform the swap
    copy_nonoverlapping_memory(t, &*x, 1);
    copy_memory(x, &*y, 1); // `x` and `y` may overlap
    copy_nonoverlapping_memory(y, &*t, 1);

    // y and t now point to the same thing, but we need to completely forget `tmp`
    // because it's no longer relevant.
    cast::forget(tmp);
}

/**
 * Replace the value at a mutable location with a new one, returning the old
 * value, without deinitialising either.
 */
#[inline]
pub unsafe fn replace<T>(dest: *mut T, mut src: T) -> T {
    mem::swap(cast::transmute(dest), &mut src); // cannot overlap
    src
}

/**
 * Reads the value from `*src` and returns it.
 */
#[inline(always)]
pub unsafe fn read<T>(src: *T) -> T {
    let mut tmp: T = mem::uninit();
    copy_nonoverlapping_memory(&mut tmp, src, 1);
    tmp
}

/**
 * Reads the value from `*src` and nulls it out.
 * This currently prevents destructors from executing.
 */
#[inline(always)]
pub unsafe fn read_and_zero<T>(dest: *mut T) -> T {
    // Copy the data out from `dest`:
    let tmp = read(&*dest);

    // Now zero out `dest`:
    zero_memory(dest, 1);

    tmp
}

/**
  Given a **T (pointer to an array of pointers),
  iterate through each *T, up to the provided `len`,
  passing to the provided callback function

  SAFETY NOTE: Pointer-arithmetic. Dragons be here.
*/
pub unsafe fn array_each_with_len<T>(arr: **T, len: uint, cb: |*T|) {
    debug!("array_each_with_len: before iterate");
    if arr.is_null() {
        fail!("ptr::array_each_with_len failure: arr input is null pointer");
    }
    //let start_ptr = *arr;
    for e in range(0, len) {
        let n = arr.offset(e as int);
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
pub unsafe fn array_each<T>(arr: **T, cb: |*T|) {
    if arr.is_null()  {
        fail!("ptr::array_each_with_len failure: arr input is null pointer");
    }
    let len = buf_len(arr);
    debug!("array_each inferred len: {}", len);
    array_each_with_len(arr, len, cb);
}

/// Extension methods for raw pointers.
pub trait RawPtr<T> {
    /// Returns the null pointer.
    fn null() -> Self;
    /// Returns true if the pointer is equal to the null pointer.
    fn is_null(&self) -> bool;
    /// Returns true if the pointer is not equal to the null pointer.
    fn is_not_null(&self) -> bool { !self.is_null() }
    /// Returns the value of this pointer (ie, the address it points to)
    fn to_uint(&self) -> uint;
    /// Returns `None` if the pointer is null, or else returns the value wrapped
    /// in `Some`.
    ///
    /// # Safety Notes
    ///
    /// While this method is useful for null-safety, it is important to note
    /// that this is still an unsafe operation because the returned value could
    /// be pointing to invalid memory.
    unsafe fn to_option(&self) -> Option<&T>;
    /// Calculates the offset from a pointer. The offset *must* be in-bounds of
    /// the object, or one-byte-past-the-end.
    unsafe fn offset(self, count: int) -> Self;
}

impl<T> RawPtr<T> for *T {
    #[inline]
    fn null() -> *T { null() }

    #[inline]
    fn is_null(&self) -> bool { *self == RawPtr::null() }

    #[inline]
    fn to_uint(&self) -> uint { *self as uint }

    #[inline]
    unsafe fn offset(self, count: int) -> *T { intrinsics::offset(self, count) }

    #[inline]
    unsafe fn to_option(&self) -> Option<&T> {
        if self.is_null() {
            None
        } else {
            Some(cast::transmute(*self))
        }
    }
}

impl<T> RawPtr<T> for *mut T {
    #[inline]
    fn null() -> *mut T { mut_null() }

    #[inline]
    fn is_null(&self) -> bool { *self == RawPtr::null() }

    #[inline]
    fn to_uint(&self) -> uint { *self as uint }

    #[inline]
    unsafe fn offset(self, count: int) -> *mut T { intrinsics::offset(self as *T, count) as *mut T }

    #[inline]
    unsafe fn to_option(&self) -> Option<&T> {
        if self.is_null() {
            None
        } else {
            Some(cast::transmute(*self))
        }
    }
}

// Equality for pointers
#[cfg(not(test))]
impl<T> Eq for *T {
    #[inline]
    fn eq(&self, other: &*T) -> bool {
        *self == *other
    }
    #[inline]
    fn ne(&self, other: &*T) -> bool { !self.eq(other) }
}

#[cfg(not(test))]
impl<T> Eq for *mut T {
    #[inline]
    fn eq(&self, other: &*mut T) -> bool {
        *self == *other
    }
    #[inline]
    fn ne(&self, other: &*mut T) -> bool { !self.eq(other) }
}

// Equivalence for pointers
#[cfg(not(test))]
impl<T> Equiv<*mut T> for *T {
    fn equiv(&self, other: &*mut T) -> bool {
        self.to_uint() == other.to_uint()
    }
}

#[cfg(not(test))]
impl<T> Equiv<*T> for *mut T {
    fn equiv(&self, other: &*T) -> bool {
        self.to_uint() == other.to_uint()
    }
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
impl<T> Ord for *T {
    #[inline]
    fn lt(&self, other: &*T) -> bool {
        *self < *other
    }
    #[inline]
    fn le(&self, other: &*T) -> bool {
        *self <= *other
    }
    #[inline]
    fn ge(&self, other: &*T) -> bool {
        *self >= *other
    }
    #[inline]
    fn gt(&self, other: &*T) -> bool {
        *self > *other
    }
}

#[cfg(not(test))]
impl<T> Ord for *mut T {
    #[inline]
    fn lt(&self, other: &*mut T) -> bool {
        *self < *other
    }
    #[inline]
    fn le(&self, other: &*mut T) -> bool {
        *self <= *other
    }
    #[inline]
    fn ge(&self, other: &*mut T) -> bool {
        *self >= *other
    }
    #[inline]
    fn gt(&self, other: &*mut T) -> bool {
        *self > *other
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
    use vec::{ImmutableVector, MutableVector};

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

            copy_memory(v1.as_mut_ptr().offset(1),
                        v0.as_ptr().offset(1), 1);
            assert!((v1[0] == 0u16 && v1[1] == 32001u16 && v1[2] == 0u16));
            copy_memory(v1.as_mut_ptr(),
                        v0.as_ptr().offset(2), 1);
            assert!((v1[0] == 32002u16 && v1[1] == 32001u16 &&
                     v1[2] == 0u16));
            copy_memory(v1.as_mut_ptr().offset(2),
                        v0.as_ptr(), 1u);
            assert!((v1[0] == 32002u16 && v1[1] == 32001u16 &&
                     v1[2] == 32000u16));
        }
    }

    #[test]
    fn test_position() {
        use libc::c_char;

        "hello".with_c_str(|p| {
            unsafe {
                assert!(2u == position(p, |c| *c == 'l' as c_char));
                assert!(4u == position(p, |c| *c == 'o' as c_char));
                assert!(5u == position(p, |c| *c == 0 as c_char));
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

        let q = unsafe { p.offset(1) };
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

            let mut ctr = 0;
            let mut iteration_count = 0;
            array_each_with_len(arr.as_ptr(), arr.len(), |e| {
                    let actual = str::raw::from_c_str(e);
                    let expected = expected_arr[ctr].with_ref(|buf| {
                            str::raw::from_c_str(buf)
                        });
                    debug!(
                        "test_ptr_array_each_with_len e: {}, a: {}",
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

            let arr_ptr = arr.as_ptr();
            let mut ctr = 0;
            let mut iteration_count = 0;
            array_each(arr_ptr, |e| {
                    let actual = str::raw::from_c_str(e);
                    let expected = expected_arr[ctr].with_ref(|buf| {
                        str::raw::from_c_str(buf)
                    });
                    debug!(
                        "test_ptr_array_each e: {}, a: {}",
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
        let ptr = xs.as_mut_ptr();
        unsafe { set_memory(ptr, 5u8, xs.len()); }
        assert_eq!(xs, [5u8, ..20]);
    }
}
