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

struct RcBox<T> {
    value: T,
    count: uint
}

/// Immutable reference counted pointer type
pub struct Rc<T> {
    priv ptr: *mut RcBox<T>,
    priv non_owned: Option<@()> // FIXME: #5601: replace with `#[non_owned]`
}

pub impl<'self, T: Owned> Rc<T> {
    fn new(value: T) -> Rc<T> {
        unsafe {
            let ptr = malloc(sys::size_of::<RcBox<T>>() as size_t) as *mut RcBox<T>;
            assert!(!ptr::is_null(ptr));
            intrinsics::move_val_init(&mut *ptr, RcBox{value: value, count: 1});
            Rc{ptr: ptr, non_owned: None}
        }
    }

    #[inline(always)]
    fn borrow(&self) -> &'self T {
        unsafe { cast::transmute_region(&(*self.ptr).value) }
    }
}

#[unsafe_destructor]
impl<T: Owned> Drop for Rc<T> {
    fn finalize(&self) {
        unsafe {
            (*self.ptr).count -= 1;
            if (*self.ptr).count == 0 {
                let mut x = intrinsics::init();
                x <-> *self.ptr;
                free(self.ptr as *c_void)
            }
        }
    }
}

impl<T: Owned> Clone for Rc<T> {
    #[inline]
    fn clone(&self) -> Rc<T> {
        unsafe {
            (*self.ptr).count += 1;
            Rc{ptr: self.ptr, non_owned: None}
        }
    }
}

#[cfg(test)]
mod test_rc {
    use super::*;

    #[test]
    fn test_simple() {
        let x = Rc::new(5);
        assert_eq!(*x.borrow(), 5);
    }

    #[test]
    fn test_clone() {
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
extern "rust-intrinsic" mod rusti {
    fn init<T>() -> T;
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
pub struct RcMut<T> {
    priv ptr: *mut RcMutBox<T>,
    priv non_owned: Option<@mut ()> // FIXME: #5601: replace with `#[non_owned]` and `#[non_const]`
}

pub impl<'self, T: Owned> RcMut<T> {
    fn new(value: T) -> RcMut<T> {
        unsafe {
            let ptr = malloc(sys::size_of::<RcMutBox<T>>() as size_t) as *mut RcMutBox<T>;
            assert!(!ptr::is_null(ptr));
            intrinsics::move_val_init(&mut *ptr, RcMutBox{value: value, count: 1, borrow: Nothing});
            RcMut{ptr: ptr, non_owned: None}
        }
    }

    /// Fails if there is already a mutable borrow of the box
    #[inline]
    fn with_borrow(&self, f: &fn(&T)) {
        unsafe {
            assert!((*self.ptr).borrow != Mutable);
            let previous = (*self.ptr).borrow;
            (*self.ptr).borrow = Immutable;
            f(cast::transmute_region(&(*self.ptr).value));
            (*self.ptr).borrow = previous;
        }
    }

    /// Fails if there is already a mutable or immutable borrow of the box
    #[inline]
    fn with_mut_borrow(&self, f: &fn(&mut T)) {
        unsafe {
            assert!((*self.ptr).borrow == Nothing);
            (*self.ptr).borrow = Mutable;
            f(cast::transmute_mut_region(&mut (*self.ptr).value));
            (*self.ptr).borrow = Nothing;
        }
    }
}

#[unsafe_destructor]
impl<T: Owned> Drop for RcMut<T> {
    fn finalize(&self) {
        unsafe {
            (*self.ptr).count -= 1;
            if (*self.ptr).count == 0 {
                let mut x = rusti::init();
                x <-> *self.ptr;
                free(self.ptr as *c_void)
            }
        }
    }
}

impl<T: Owned> Clone for RcMut<T> {
    #[inline]
    fn clone(&self) -> RcMut<T> {
        unsafe {
            (*self.ptr).count += 1;
            RcMut{ptr: self.ptr, non_owned: None}
        }
    }
}

#[cfg(test)]
mod test_rc_mut {
    use super::*;

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
