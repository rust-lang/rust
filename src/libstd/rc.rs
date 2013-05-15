// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

/** Task-local reference counted smart pointers

Task-local reference counted smart pointers are an alternative to managed boxes with deterministic
destruction. They are restricted to containing `Owned` types in order to prevent cycles.

*/

use core::libc::{c_void, size_t, malloc, free};
use core::unstable::intrinsics;
use core::util;

struct RcBox<T> {
    value: T,
    count: uint
}

/// Immutable reference counted pointer type
#[non_owned]
pub struct Rc<T> {
    priv ptr: *mut RcBox<T>,
}

pub impl<T: Owned> Rc<T> {
    fn new(value: T) -> Rc<T> {
        unsafe {
            let ptr = malloc(sys::size_of::<RcBox<T>>() as size_t) as *mut RcBox<T>;
            assert!(!ptr::is_null(ptr));
            intrinsics::move_val_init(&mut *ptr, RcBox{value: value, count: 1});
            Rc{ptr: ptr}
        }
    }

    #[inline(always)]
    fn borrow<'r>(&'r self) -> &'r T {
        unsafe { cast::copy_lifetime(self, &(*self.ptr).value) }
    }
}

#[unsafe_destructor]
#[cfg(not(stage0))]
impl<T: Owned> Drop for Rc<T> {
    fn finalize(&self) {
        unsafe {
            (*self.ptr).count -= 1;
            if (*self.ptr).count == 0 {
                util::replace_ptr(self.ptr, intrinsics::uninit());
                free(self.ptr as *c_void)
            }
        }
    }
}

#[unsafe_destructor]
#[cfg(stage0)]
impl<T: Owned> Drop for Rc<T> {
    fn finalize(&self) {
        unsafe {
            (*self.ptr).count -= 1;
            if (*self.ptr).count == 0 {
                util::replace_ptr(self.ptr, intrinsics::init());
                free(self.ptr as *c_void)
            }
        }
    }
}


impl<T: Owned> Clone for Rc<T> {
    /// Return a shallow copy of the reference counted pointer.
    #[inline]
    fn clone(&self) -> Rc<T> {
        unsafe {
            (*self.ptr).count += 1;
            Rc{ptr: self.ptr}
        }
    }
}

impl<T: Owned + DeepClone> DeepClone for Rc<T> {
    /// Return a deep copy of the reference counted pointer.
    #[inline]
    fn deep_clone(&self) -> Rc<T> {
        Rc::new(self.borrow().deep_clone())
    }
}

#[cfg(test)]
mod test_rc {
    use super::*;
    use core::cell::Cell;

    #[test]
    fn test_clone() {
        let x = Rc::new(Cell(5));
        let y = x.clone();
        do x.borrow().with_mut_ref |inner| {
            *inner = 20;
        }
        assert_eq!(y.borrow().take(), 20);
    }

    #[test]
    fn test_deep_clone() {
        let x = Rc::new(Cell(5));
        let y = x.deep_clone();
        do x.borrow().with_mut_ref |inner| {
            *inner = 20;
        }
        assert_eq!(y.borrow().take(), 5);
    }

    #[test]
    fn test_simple() {
        let x = Rc::new(5);
        assert_eq!(*x.borrow(), 5);
    }

    #[test]
    fn test_simple_clone() {
        let x = Rc::new(5);
        let y = x.clone();
        assert_eq!(*x.borrow(), 5);
        assert_eq!(*y.borrow(), 5);
    }

    #[test]
    fn test_destructor() {
        let x = Rc::new(~5);
        assert_eq!(**x.borrow(), 5);
    }
}

#[abi = "rust-intrinsic"]
extern "rust-intrinsic" {
    fn init<T>() -> T;
    #[cfg(not(stage0))]
    fn uninit<T>() -> T;
}

#[deriving(Eq)]
enum Borrow {
    Mutable,
    Immutable,
    Nothing
}

struct RcMutBox<T> {
    value: T,
    count: uint,
    borrow: Borrow
}

/// Mutable reference counted pointer type
#[non_owned]
#[mutable]
pub struct RcMut<T> {
    priv ptr: *mut RcMutBox<T>,
}

pub impl<T: Owned> RcMut<T> {
    fn new(value: T) -> RcMut<T> {
        unsafe {
            let ptr = malloc(sys::size_of::<RcMutBox<T>>() as size_t) as *mut RcMutBox<T>;
            assert!(!ptr::is_null(ptr));
            intrinsics::move_val_init(&mut *ptr, RcMutBox{value: value, count: 1, borrow: Nothing});
            RcMut{ptr: ptr}
        }
    }

    /// Fails if there is already a mutable borrow of the box
    #[inline]
    fn with_borrow<U>(&self, f: &fn(&T) -> U) -> U {
        unsafe {
            assert!((*self.ptr).borrow != Mutable);
            let previous = (*self.ptr).borrow;
            (*self.ptr).borrow = Immutable;
            let res = f(&(*self.ptr).value);
            (*self.ptr).borrow = previous;
            res
        }
    }

    /// Fails if there is already a mutable or immutable borrow of the box
    #[inline]
    fn with_mut_borrow<U>(&self, f: &fn(&mut T) -> U) -> U {
        unsafe {
            assert!((*self.ptr).borrow == Nothing);
            (*self.ptr).borrow = Mutable;
            let res = f(&mut (*self.ptr).value);
            (*self.ptr).borrow = Nothing;
            res
        }
    }
}

#[unsafe_destructor]
#[cfg(not(stage0))]
impl<T: Owned> Drop for RcMut<T> {
    fn finalize(&self) {
        unsafe {
            (*self.ptr).count -= 1;
            if (*self.ptr).count == 0 {
                util::replace_ptr(self.ptr, uninit());
                free(self.ptr as *c_void)
            }
        }
    }
}

#[unsafe_destructor]
#[cfg(stage0)]
impl<T: Owned> Drop for RcMut<T> {
    fn finalize(&self) {
        unsafe {
            (*self.ptr).count -= 1;
            if (*self.ptr).count == 0 {
                util::replace_ptr(self.ptr, init());
                free(self.ptr as *c_void)
            }
        }
    }
}

impl<T: Owned> Clone for RcMut<T> {
    /// Return a shallow copy of the reference counted pointer.
    #[inline]
    fn clone(&self) -> RcMut<T> {
        unsafe {
            (*self.ptr).count += 1;
            RcMut{ptr: self.ptr}
        }
    }
}

impl<T: Owned + DeepClone> DeepClone for RcMut<T> {
    /// Return a deep copy of the reference counted pointer.
    #[inline]
    fn deep_clone(&self) -> RcMut<T> {
        do self.with_borrow |x| {
            // FIXME: #6497: should avoid freeze (slow)
            RcMut::new(x.deep_clone())
        }
    }
}

#[cfg(test)]
mod test_rc_mut {
    use super::*;

    #[test]
    fn test_clone() {
        let x = RcMut::new(5);
        let y = x.clone();
        do x.with_mut_borrow |value| {
            *value = 20;
        }
        do y.with_borrow |value| {
            assert_eq!(*value, 20);
        }
    }

    #[test]
    fn test_deep_clone() {
        let x = RcMut::new(5);
        let y = x.deep_clone();
        do x.with_mut_borrow |value| {
            *value = 20;
        }
        do y.with_borrow |value| {
            assert_eq!(*value, 5);
        }
    }

    #[test]
    fn borrow_many() {
        let x = RcMut::new(5);
        let y = x.clone();

        do x.with_borrow |a| {
            assert_eq!(*a, 5);
            do y.with_borrow |b| {
                assert_eq!(*b, 5);
                do x.with_borrow |c| {
                    assert_eq!(*c, 5);
                }
            }
        }
    }

    #[test]
    fn modify() {
        let x = RcMut::new(5);
        let y = x.clone();

        do y.with_mut_borrow |a| {
            assert_eq!(*a, 5);
            *a = 6;
        }

        do x.with_borrow |a| {
            assert_eq!(*a, 6);
        }
    }

    #[test]
    fn release_immutable() {
        let x = RcMut::new(5);
        do x.with_borrow |_| {}
        do x.with_mut_borrow |_| {}
    }

    #[test]
    fn release_mutable() {
        let x = RcMut::new(5);
        do x.with_mut_borrow |_| {}
        do x.with_borrow |_| {}
    }

    #[test]
    #[should_fail]
    fn frozen() {
        let x = RcMut::new(5);
        let y = x.clone();

        do x.with_borrow |_| {
            do y.with_mut_borrow |_| {
            }
        }
    }

    #[test]
    #[should_fail]
    fn mutable_dupe() {
        let x = RcMut::new(5);
        let y = x.clone();

        do x.with_mut_borrow |_| {
            do y.with_mut_borrow |_| {
            }
        }
    }

    #[test]
    #[should_fail]
    fn mutable_freeze() {
        let x = RcMut::new(5);
        let y = x.clone();

        do x.with_mut_borrow |_| {
            do y.with_borrow |_| {
            }
        }
    }

    #[test]
    #[should_fail]
    fn restore_freeze() {
        let x = RcMut::new(5);
        let y = x.clone();

        do x.with_borrow |_| {
            do x.with_borrow |_| {}
            do y.with_mut_borrow |_| {}
        }
    }
}
