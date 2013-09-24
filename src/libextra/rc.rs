// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#[allow(missing_doc)];

/** Task-local reference counted smart pointers

Task-local reference counted smart pointers are an alternative to managed boxes
with deterministic destruction. They are restricted to containing types that
are either `Send` or `Freeze` (or both) to prevent cycles.

Neither `Rc<T>` is never `Send` and `Mut<T>` is never `Freeze`. If `T` is `Freeze`, a
cycle cannot be created with `Rc<T>` because there is no way to modify it after creation.

`Rc<Mut<U>>` requires `U` to be `Send` so that no cycles are created.

*/


use std::cast;
use std::mutable::Mut;
use std::ptr;
use std::unstable::intrinsics;

// Convert ~T into *mut T without dropping it
#[inline]
unsafe fn owned_to_raw<T>(mut box: ~T) -> *mut T {
    let ptr = ptr::to_mut_unsafe_ptr(box);
    intrinsics::forget(box);
    ptr
}

struct RcBox<T> {
    value: T,
    count: uint
}

/// Immutable reference counted pointer type
#[unsafe_no_drop_flag]
#[no_send]
pub struct Rc<T> {
    priv ptr: *mut RcBox<T>,
}

impl<T> Rc<T> {
    unsafe fn new(value: T) -> Rc<T> {
        Rc{ptr: owned_to_raw(~RcBox{value: value, count: 1})}
    }
}

impl<T: Send> Rc<T> {
    pub fn from_send(value: T) -> Rc<T> {
        unsafe { Rc::new(value) }
    }

    pub fn from_send_mut(value: T) -> Rc<Mut<T>> {
        unsafe { Rc::new(Mut::new(value)) }
    }
}

impl<T: Freeze> Rc<T> {
    pub fn from_freeze(value: T) -> Rc<T> {
        unsafe { Rc::new(value) }
    }
}

impl<T> Rc<T> {
    #[inline]
    pub fn borrow<'r>(&'r self) -> &'r T {
        unsafe { cast::copy_lifetime(self, &(*self.ptr).value) }
    }
}

#[unsafe_destructor]
impl<T> Drop for Rc<T> {
    fn drop(&mut self) {
        unsafe {
            if self.ptr.is_not_null() {
                (*self.ptr).count -= 1;
                if (*self.ptr).count == 0 {
                    let _: ~T = cast::transmute(self.ptr);
                }
            }
        }
    }
}

impl<T> Clone for Rc<T> {
    /// Return a shallow copy of the reference counted pointer.
    #[inline]
    fn clone(&self) -> Rc<T> {
        unsafe {
            (*self.ptr).count += 1;
            Rc{ptr: self.ptr}
        }
    }
}

impl<T: DeepClone> DeepClone for Rc<T> {
    /// Return a deep copy of the reference counted pointer.
    #[inline]
    fn deep_clone(&self) -> Rc<T> {
        unsafe { Rc::new(self.borrow().deep_clone()) }
    }
}

#[cfg(test)]
mod test_rc {
    use super::*;

    #[test]
    fn test_clone() {
        let x = Rc::from_send_mut(5);
        let y = x.clone();
        do x.borrow().map_mut |inner| {
            *inner = 20;
        }
        assert_eq!(y.borrow().map(|t| *t), 20);
    }

    #[test]
    fn test_deep_clone() {
        let x = Rc::from_send_mut(5);
        let y = x.deep_clone();
        do x.borrow().map_mut |inner| {
            *inner = 20;
        }
        assert_eq!(y.borrow().map(|t| *t), 5);
    }

    #[test]
    fn test_simple() {
        let x = Rc::from_freeze(5);
        assert_eq!(*x.borrow(), 5);
    }

    #[test]
    fn test_simple_clone() {
        let x = Rc::from_freeze(5);
        let y = x.clone();
        assert_eq!(*x.borrow(), 5);
        assert_eq!(*y.borrow(), 5);
    }

    #[test]
    fn test_destructor() {
        let x = Rc::from_send(~5);
        assert_eq!(**x.borrow(), 5);
    }
}

#[cfg(test)]
mod test_rc_mut {
    use super::*;

    #[test]
    fn test_clone() {
        let x = Rc::from_send_mut(5);
        let y = x.clone();
        do x.borrow().map_mut |value| {
            *value = 20;
        }
        do y.borrow().map_mut |value| {
            assert_eq!(*value, 20);
        }
    }

    #[test]
    fn test_deep_clone() {
        let x = Rc::from_send_mut(5);
        let y = x.deep_clone();
        do x.borrow().map_mut |value| {
            *value = 20;
        }
        do y.borrow().map |value| {
            assert_eq!(*value, 5);
        }
    }

    #[test]
    fn borrow_many() {
        let x = Rc::from_send_mut(5);
        let y = x.clone();

        do x.borrow().map |a| {
            assert_eq!(*a, 5);
            do y.borrow().map |b| {
                assert_eq!(*b, 5);
                do x.borrow().map |c| {
                    assert_eq!(*c, 5);
                }
            }
        }
    }

    #[test]
    fn modify() {
        let x = Rc::from_send_mut(5);
        let y = x.clone();

        do y.borrow().map_mut |a| {
            assert_eq!(*a, 5);
            *a = 6;
        }

        do x.borrow().map |a| {
            assert_eq!(*a, 6);
        }
    }

    #[test]
    fn release_immutable() {
        let x = Rc::from_send_mut(5);
        do x.borrow().map |_| {}
        do x.borrow().map_mut |_| {}
    }

    #[test]
    fn release_mutable() {
        let x = Rc::from_send_mut(5);
        do x.borrow().map_mut |_| {}
        do x.borrow().map |_| {}
    }

    #[test]
    #[should_fail]
    fn frozen() {
        let x = Rc::from_send_mut(5);
        let y = x.clone();

        do x.borrow().map |_| {
            do y.borrow().map_mut |_| {
            }
        }
    }

    #[test]
    #[should_fail]
    fn mutable_dupe() {
        let x = Rc::from_send_mut(5);
        let y = x.clone();

        do x.borrow().map_mut |_| {
            do y.borrow().map_mut |_| {
            }
        }
    }

    #[test]
    #[should_fail]
    fn mutable_freeze() {
        let x = Rc::from_send_mut(5);
        let y = x.clone();

        do x.borrow().map_mut |_| {
            do y.borrow().map |_| {
            }
        }
    }

    #[test]
    #[should_fail]
    fn restore_freeze() {
        let x = Rc::from_send_mut(5);
        let y = x.clone();

        do x.borrow().map |_| {
            do x.borrow().map |_| {}
            do y.borrow().map_mut |_| {}
        }
    }
}
