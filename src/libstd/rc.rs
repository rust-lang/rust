// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

/*! Task-local reference counted boxes

The `Rc` type provides shared ownership of an immutable value. Destruction is deterministic, and
will occur as soon as the last owner is gone. It is marked as non-sendable because it avoids the
overhead of atomic reference counting.

The `RcMut` type provides shared ownership of a mutable value. Since multiple owners prevent
inherited mutability, a dynamic freezing check is used to maintain the invariant that an `&mut`
reference is a unique handle and the type is marked as non-`Freeze`.

*/

use ptr::RawPtr;
use unstable::intrinsics::transmute;
use ops::Drop;
use kinds::{Freeze, Send};
use clone::{Clone, DeepClone};

struct RcBox<T> {
    value: T,
    count: uint
}

/// Immutable reference counted pointer type
#[unsafe_no_drop_flag]
#[no_send]
pub struct Rc<T> {
    priv ptr: *mut RcBox<T>
}

impl<T: Freeze> Rc<T> {
    /// Construct a new reference-counted box from a `Freeze` value
    #[inline]
    pub fn new(value: T) -> Rc<T> {
        unsafe {
            Rc::new_unchecked(value)
        }
    }
}

impl<T> Rc<T> {
    /// Unsafety construct a new reference-counted box from any value.
    ///
    /// If the type is not `Freeze`, the `Rc` box will incorrectly still be considered as a `Freeze`
    /// type. It is also possible to create cycles, which will leak, and may interact poorly with
    /// managed pointers.
    #[inline]
    pub unsafe fn new_unchecked(value: T) -> Rc<T> {
        Rc{ptr: transmute(~RcBox{value: value, count: 1})}
    }

    /// Borrow the value contained in the reference-counted box
    #[inline]
    pub fn borrow<'r>(&'r self) -> &'r T {
        unsafe { &(*self.ptr).value }
    }
}

impl<T> Clone for Rc<T> {
    #[inline]
    fn clone(&self) -> Rc<T> {
        unsafe {
            (*self.ptr).count += 1;
            Rc{ptr: self.ptr}
        }
    }
}

impl<T: DeepClone> DeepClone for Rc<T> {
    #[inline]
    fn deep_clone(&self) -> Rc<T> {
        unsafe { Rc::new_unchecked(self.borrow().deep_clone()) }
    }
}

#[unsafe_destructor]
impl<T> Drop for Rc<T> {
    fn drop(&mut self) {
        unsafe {
            if self.ptr.is_not_null() {
                (*self.ptr).count -= 1;
                if (*self.ptr).count == 0 {
                    let _: ~RcBox<T> = transmute(self.ptr);
                }
            }
        }
    }
}

#[cfg(test)]
mod test_rc {
    use super::*;
    use cell::Cell;

    #[test]
    fn test_clone() {
        unsafe {
            let x = Rc::new_unchecked(Cell::new(5));
            let y = x.clone();
            do x.borrow().with_mut_ref |inner| {
                *inner = 20;
            }
            assert_eq!(y.borrow().take(), 20);
        }
    }

    #[test]
    fn test_deep_clone() {
        unsafe {
            let x = Rc::new_unchecked(Cell::new(5));
            let y = x.deep_clone();
            do x.borrow().with_mut_ref |inner| {
                *inner = 20;
            }
            assert_eq!(y.borrow().take(), 5);
        }
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
        unsafe {
            let x = Rc::new_unchecked(~5);
            assert_eq!(**x.borrow(), 5);
        }
    }
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
#[no_send]
#[no_freeze]
#[unsafe_no_drop_flag]
pub struct RcMut<T> {
    priv ptr: *mut RcMutBox<T>,
}

impl<T: Freeze> RcMut<T> {
    /// Construct a new mutable reference-counted box from a `Freeze` value
    #[inline]
    pub fn new(value: T) -> RcMut<T> {
        unsafe { RcMut::new_unchecked(value) }
    }
}

impl<T: Send> RcMut<T> {
    /// Construct a new mutable reference-counted box from a `Send` value
    #[inline]
    pub fn from_send(value: T) -> RcMut<T> {
        unsafe { RcMut::new_unchecked(value) }
    }
}

impl<T> RcMut<T> {
    /// Unsafety construct a new mutable reference-counted box from any value.
    ///
    /// It is possible to create cycles, which will leak, and may interact
    /// poorly with managed pointers.
    #[inline]
    pub unsafe fn new_unchecked(value: T) -> RcMut<T> {
        RcMut{ptr: transmute(~RcMutBox{value: value, count: 1, borrow: Nothing})}
    }
}

impl<T> RcMut<T> {
    /// Fails if there is already a mutable borrow of the box
    #[inline]
    pub fn with_borrow<U>(&self, f: &fn(&T) -> U) -> U {
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
    pub fn with_mut_borrow<U>(&self, f: &fn(&mut T) -> U) -> U {
        unsafe {
            assert_eq!((*self.ptr).borrow, Nothing);
            (*self.ptr).borrow = Mutable;
            let res = f(&mut (*self.ptr).value);
            (*self.ptr).borrow = Nothing;
            res
        }
    }
}

#[unsafe_destructor]
impl<T> Drop for RcMut<T> {
    fn drop(&mut self) {
        unsafe {
            if self.ptr.is_not_null() {
                (*self.ptr).count -= 1;
                if (*self.ptr).count == 0 {
                    let _: ~RcMutBox<T> = transmute(self.ptr);
                }
            }
        }
    }
}

impl<T> Clone for RcMut<T> {
    /// Return a shallow copy of the reference counted pointer.
    #[inline]
    fn clone(&self) -> RcMut<T> {
        unsafe {
            (*self.ptr).count += 1;
            RcMut{ptr: self.ptr}
        }
    }
}

impl<T: DeepClone> DeepClone for RcMut<T> {
    /// Return a deep copy of the reference counted pointer.
    #[inline]
    fn deep_clone(&self) -> RcMut<T> {
        do self.with_borrow |x| {
            // FIXME: #6497: should avoid freeze (slow)
            unsafe { RcMut::new_unchecked(x.deep_clone()) }
        }
    }
}

#[cfg(test)]
mod test_rc_mut {
    use super::*;

    #[test]
    fn test_clone() {
        let x = RcMut::from_send(5);
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
        let x = RcMut::from_send(5);
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
        let x = RcMut::from_send(5);
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
        let x = RcMut::from_send(5);
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
        let x = RcMut::from_send(5);
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
