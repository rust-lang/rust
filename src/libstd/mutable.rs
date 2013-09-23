// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! A mutable memory location with dynamically checked borrow rules.

use cast;
use cmp::Eq;
use util;

use ops::Drop;
use clone::{Clone, DeepClone};
use option::{Option, Some};

/// A mutable slot with dynamically checked borrow rules.
#[no_freeze]
pub struct Mut<T> {
    priv value: T,
    priv status: BorrowFlag,
    priv nc: util::NonCopyable,
}

enum BorrowFlag {
    /// the Reading count increases with each ReadPtr,
    /// starting with 0 for the first one so that all values are used.
    Reading(uint),
    Writing,
    Unused,
}


impl<T> Mut<T> {
    /// Create a new `Mut` containing `value`
    #[inline]
    pub fn new(value: T) -> Mut<T> {
        Mut{value: value, status: Unused, nc: util::NonCopyable}
    }

    unsafe fn as_mut<'a>(&'a self) -> &'a mut Mut<T> {
        cast::transmute_mut(self)
    }

    /// Consume the Mut<T> and extract the held value
    pub fn unwrap(self) -> T {
        match self.status {
            Unused => self.value,
            // debug assert? This case should be statically prevented
            // by regular &'a self borrowing and noncopyability.
            Writing | Reading(*) => fail!("borrow inconsistency in Mut<T>"),
        }
    }

    /// Borrow the Mut<T> for immutable access through a wrapped pointer
    ///
    /// The borrow lasts until the returned `ReadPtr` expires its scope.
    /// Multiple simultaneous immutable borrows are possible.
    ///
    /// Fails if the slot is already borrowed mutably.
    #[inline]
    pub fn borrow<'a>(&'a self) -> ReadPtr<'a, T> {
        unsafe {
            let mut_self = self.as_mut();
            mut_self.status = match mut_self.status {
                Unused => Reading(0),
                Reading(n) => Reading(n + 1),
                Writing => fail!("borrow: Mut<T> reserved by borrow_mut"),
            };
        }
        ReadPtr{parent: self}
    }

    /// Borrow the Mut<T> for mutable access through a wrapped pointer.
    ///
    /// The borrow lasts until the returned `WritePtr` expires its scope.
    /// `borrow_mut` is an exclusive borrow.
    ///
    /// Fails if the slot has any outstanding borrow.
    #[inline]
    pub fn borrow_mut<'a>(&'a self) -> WritePtr<'a, T> {
        unsafe {
            let mut_self = self.as_mut();
            mut_self.status = match mut_self.status {
                Unused => Writing,
                Writing | Reading(*) => fail!("borrow_mut: Mut<T> already in use"),
            };
            WritePtr{parent: mut_self}
        }
    }

    /// Borrow the Mut<T> and apply `f`
    #[inline]
    pub fn map<U>(&self, f: &fn(&T) -> U) -> U {
        let r_ptr = self.borrow();
        f(r_ptr.get())
    }

    /// Borrow the Mut<T> mutably and apply `f`
    #[inline]
    pub fn map_mut<U>(&self, f: &fn(&mut T) -> U) -> U {
        let mut m_ptr = self.borrow_mut();
        f(m_ptr.get())
    }
}

impl<U> Mut<Option<U>> {
    /// Create a Mut holding `Some(value)`
    #[inline]
    pub fn new_some(value: U) -> Mut<Option<U>> {
        Mut::new(Some(value))
    }

    /// Borrow the Mut<T> mutably, replace the held value with None,
    /// and return the held Option value.
    ///
    /// Fails if the Mut<T> could not be borrowed mutably.
    #[inline]
    pub fn take(&self) -> Option<U> {
        // specialize for performance
        unsafe {
            match self.status {
                Unused => self.as_mut().value.take(),
                Writing | Reading(*) => fail!("Mut::take: Mut<T> already in use"),
            }
        }
    }

    /// Borrow the Mut<T> mutably, replace the held value with None,
    /// and return the unwrapped Option value.
    ///
    /// Fails if the Mut<T> could not be borrowed mutably,
    /// or if the held value is `None`.
    #[inline]
    pub fn take_unwrap(&self) -> U {
        self.take().expect("Mut::take_unwrap: attempt to unwrap empty Mut<T>")
    }

    /// Replace the held value with `Some(value)`
    ///
    /// Fails if the Mut<T> could not be borrowed mutably,
    /// or if the held value is not `None`.
    #[inline]
    pub fn put_back(&self, value: U) {
        let mut mptr = self.borrow_mut();
        if !mptr.get().is_none() {
            fail!("Mut::put_back: already holding Some value")
        }
        *mptr.get() = Some(value);
    }
}

/// A borrowed pointer that holds its borrow
/// until it expires its scope.
pub struct ReadPtr<'self, T> {
    priv parent: &'self Mut<T>
}

impl<'self, T> ReadPtr<'self, T> {
    /// Resolve the `ReadPtr` to a borrowed pointer
    #[inline]
    pub fn get<'a>(&'a self) -> &'a T {
        &self.parent.value
    }
}

#[unsafe_destructor]
impl<'self, T> Drop for ReadPtr<'self, T> {
    fn drop(&mut self) {
        unsafe {
            let mut_par = self.parent.as_mut();
            match mut_par.status {
                Reading(0) => mut_par.status = Unused,
                Reading(n) => mut_par.status = Reading(n - 1),
                // XXX: should be debug assert?
                Writing | Unused => error!("ReadPtr::drop: borrow inconsistency in Mut<T>")
            }
        }
    }
}

/// A mutable borrowed pointer that holds its borrow
/// until it expires its scope.
pub struct WritePtr<'self, T> {
    priv parent: &'self mut Mut<T>
}

impl<'self, T> WritePtr<'self, T> {
    /// Resolve the `ReadPtr` to a mutable borrowed pointer
    #[inline]
    pub fn get<'a>(&'a mut self) -> &'a mut T {
        &mut self.parent.value
    }
}

#[unsafe_destructor]
impl<'self, T> Drop for WritePtr<'self, T> {
    fn drop(&mut self) {
        unsafe {
            let mut_par = self.parent.as_mut();
            match mut_par.status {
                Writing => mut_par.status = Unused,
                // XXX: should debug assert?
                Reading(*) | Unused => error!("WritePtr::drop: borrow inconsistency in Mut<T>")
            }
        }
    }
}

impl<T: Clone> Clone for Mut<T> {
    fn clone(&self) -> Mut<T> {
        let rptr = self.borrow();
        Mut::new(rptr.get().clone())
    }
}

impl<T: DeepClone> DeepClone for Mut<T> {
    fn deep_clone(&self) -> Mut<T> {
        let rptr = self.borrow();
        Mut::new(rptr.get().deep_clone())
    }
}

impl<T: Eq> Eq for Mut<T> {
    fn eq(&self, other: &Mut<T>) -> bool {
        let rptr = self.borrow();
        let optr = other.borrow();
        rptr.get() == optr.get()
    }
}


#[cfg(test)]
pub fn test_read_then_read_x() {
    use util::ignore;

    let obj = Mut::new(1);
    let r = obj.borrow();
    let q = obj.borrow();
    assert_eq!(r.get(), q.get());

    match obj.status { Reading(1) => (), _ => fail!() }
    ignore(r);
    match obj.status { Reading(0) => (), _ => fail!() }
    ignore(q);
    match obj.status { Unused => (), _ => fail!() }
}


#[cfg(test)]
mod tests {
    use super::*;
    use option::{Some, None};
    use util::ignore;

    #[test]
    fn test_read_then_read() {
        test_read_then_read_x()
    }

    #[test]
    #[should_fail]
    fn test_read_release_partial_then_write() {
        let obj = Mut::new(1);
        let r = obj.borrow();
        let q = obj.borrow();
        assert_eq!(r.get(), q.get());
        ignore(r);
        let _ = obj.borrow_mut();
    }

    #[test]
    #[should_fail]
    fn test_write_then_write() {
        let obj = Mut::new(1);
        let mptr = obj.borrow_mut();
        let nptr = obj.borrow_mut();
        ignore(mptr);
        ignore(nptr);
    }

    #[test]
    fn test_read_release_then_write() {
        let obj = Mut::new(1);
        let r = obj.borrow();
        let q = obj.borrow();
        assert_eq!(r.get(), q.get());
        ignore(r);
        ignore(q);
        let mut m = obj.borrow_mut();
        *m.get() = 99;
        ignore(m);
        let r = obj.borrow();
        assert_eq!(*r.get(), 99);
    }


    #[test]
    #[should_fail]
    fn test_read_then_write() {
        let obj = Mut::new(1);
        let r = obj.borrow();
        let _ = obj.borrow_mut();
        ignore(r);
    }

    #[test]
    fn test_option_take() {
        let obj = Mut::new(Some(3));
        let v = None::<int>.unwrap_or_else(|| obj.take().unwrap());
        assert_eq!(v, 3);
        assert!(obj.map(|x| x.is_none()));
    }
}
