// Copyright 2012-2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Types dealing with dynamic mutability

use cast;
use clone::Clone;
use cmp::Eq;
use fmt;
use kinds::{marker, Copy};
use ops::{Deref, DerefMut, Drop};
use option::{None, Option, Some};
use ty::Unsafe;

/// A mutable memory location that admits only `Copy` data.
pub struct Cell<T> {
    value: Unsafe<T>,
    noshare: marker::NoShare,
}

impl<T:Copy> Cell<T> {
    /// Creates a new `Cell` containing the given value.
    pub fn new(value: T) -> Cell<T> {
        Cell {
            value: Unsafe::new(value),
            noshare: marker::NoShare,
        }
    }

    /// Returns a copy of the contained value.
    #[inline]
    pub fn get(&self) -> T {
        unsafe{ *self.value.get() }
    }

    /// Sets the contained value.
    #[inline]
    pub fn set(&self, value: T) {
        unsafe {
            *self.value.get() = value;
        }
    }
}

impl<T:Copy> Clone for Cell<T> {
    fn clone(&self) -> Cell<T> {
        Cell::new(self.get())
    }
}

impl<T:Eq + Copy> Eq for Cell<T> {
    fn eq(&self, other: &Cell<T>) -> bool {
        self.get() == other.get()
    }
}

impl<T: fmt::Show> fmt::Show for Cell<T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f.buf, r"Cell \{ value: {} \}", unsafe{*&self.value.get()})
    }
}

/// A mutable memory location with dynamically checked borrow rules
pub struct RefCell<T> {
    value: Unsafe<T>,
    borrow: BorrowFlag,
    nocopy: marker::NoCopy,
    noshare: marker::NoShare,
}

// Values [1, MAX-1] represent the number of `Ref` active
// (will not outgrow its range since `uint` is the size of the address space)
type BorrowFlag = uint;
static UNUSED: BorrowFlag = 0;
static WRITING: BorrowFlag = -1;

impl<T> RefCell<T> {
    /// Create a new `RefCell` containing `value`
    pub fn new(value: T) -> RefCell<T> {
        RefCell {
            value: Unsafe::new(value),
            nocopy: marker::NoCopy,
            noshare: marker::NoShare,
            borrow: UNUSED,
        }
    }

    /// Consumes the `RefCell`, returning the wrapped value.
    pub fn unwrap(self) -> T {
        assert!(self.borrow == UNUSED);
        unsafe{self.value.unwrap()}
    }

    unsafe fn as_mut<'a>(&'a self) -> &'a mut RefCell<T> {
        cast::transmute_mut(self)
    }

    /// Attempts to immutably borrow the wrapped value.
    ///
    /// The borrow lasts until the returned `Ref` exits scope. Multiple
    /// immutable borrows can be taken out at the same time.
    ///
    /// Returns `None` if the value is currently mutably borrowed.
    pub fn try_borrow<'a>(&'a self) -> Option<Ref<'a, T>> {
        match self.borrow {
            WRITING => None,
            _ => {
                unsafe { self.as_mut().borrow += 1; }
                Some(Ref { parent: self })
            }
        }
    }

    /// Immutably borrows the wrapped value.
    ///
    /// The borrow lasts until the returned `Ref` exits scope. Multiple
    /// immutable borrows can be taken out at the same time.
    ///
    /// # Failure
    ///
    /// Fails if the value is currently mutably borrowed.
    pub fn borrow<'a>(&'a self) -> Ref<'a, T> {
        match self.try_borrow() {
            Some(ptr) => ptr,
            None => fail!("RefCell<T> already mutably borrowed")
        }
    }

    /// Mutably borrows the wrapped value.
    ///
    /// The borrow lasts until the returned `RefMut` exits scope. The value
    /// cannot be borrowed while this borrow is active.
    ///
    /// Returns `None` if the value is currently borrowed.
    pub fn try_borrow_mut<'a>(&'a self) -> Option<RefMut<'a, T>> {
        match self.borrow {
            UNUSED => unsafe {
                let mut_self = self.as_mut();
                mut_self.borrow = WRITING;
                Some(RefMut { parent: mut_self })
            },
            _ => None
        }
    }

    /// Mutably borrows the wrapped value.
    ///
    /// The borrow lasts until the returned `RefMut` exits scope. The value
    /// cannot be borrowed while this borrow is active.
    ///
    /// # Failure
    ///
    /// Fails if the value is currently borrowed.
    pub fn borrow_mut<'a>(&'a self) -> RefMut<'a, T> {
        match self.try_borrow_mut() {
            Some(ptr) => ptr,
            None => fail!("RefCell<T> already borrowed")
        }
    }

    /// Sets the value, replacing what was there.
    ///
    /// # Failure
    ///
    /// Fails if the value is currently borrowed.
    #[inline]
    pub fn set(&self, value: T) {
        *self.borrow_mut() = value;
    }
}

impl<T:Clone> RefCell<T> {
    /// Returns a copy of the contained value.
    ///
    /// # Failure
    ///
    /// Fails if the value is currently mutably borrowed.
    #[inline]
    pub fn get(&self) -> T {
        (*self.borrow()).clone()
    }
}

impl<T: Clone> Clone for RefCell<T> {
    fn clone(&self) -> RefCell<T> {
        RefCell::new(self.get())
    }
}

impl<T: Eq> Eq for RefCell<T> {
    fn eq(&self, other: &RefCell<T>) -> bool {
        *self.borrow() == *other.borrow()
    }
}

/// Wraps a borrowed reference to a value in a `RefCell` box.
pub struct Ref<'b, T> {
    parent: &'b RefCell<T>
}

#[unsafe_destructor]
impl<'b, T> Drop for Ref<'b, T> {
    fn drop(&mut self) {
        assert!(self.parent.borrow != WRITING && self.parent.borrow != UNUSED);
        unsafe { self.parent.as_mut().borrow -= 1; }
    }
}

impl<'b, T> Deref<T> for Ref<'b, T> {
    #[inline]
    fn deref<'a>(&'a self) -> &'a T {
        unsafe{ &*self.parent.value.get() }
    }
}

/// Wraps a mutable borrowed reference to a value in a `RefCell` box.
pub struct RefMut<'b, T> {
    parent: &'b mut RefCell<T>
}

#[unsafe_destructor]
impl<'b, T> Drop for RefMut<'b, T> {
    fn drop(&mut self) {
        assert!(self.parent.borrow == WRITING);
        self.parent.borrow = UNUSED;
    }
}

impl<'b, T> Deref<T> for RefMut<'b, T> {
    #[inline]
    fn deref<'a>(&'a self) -> &'a T {
        unsafe{ &*self.parent.value.get() }
    }
}

impl<'b, T> DerefMut<T> for RefMut<'b, T> {
    #[inline]
    fn deref_mut<'a>(&'a mut self) -> &'a mut T {
        unsafe{ &mut *self.parent.value.get() }
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn smoketest_cell() {
        let x = Cell::new(10);
        assert_eq!(x, Cell::new(10));
        assert_eq!(x.get(), 10);
        x.set(20);
        assert_eq!(x, Cell::new(20));
        assert_eq!(x.get(), 20);

        let y = Cell::new((30, 40));
        assert_eq!(y, Cell::new((30, 40)));
        assert_eq!(y.get(), (30, 40));
    }

    #[test]
    fn double_imm_borrow() {
        let x = RefCell::new(0);
        let _b1 = x.borrow();
        x.borrow();
    }

    #[test]
    fn no_mut_then_imm_borrow() {
        let x = RefCell::new(0);
        let _b1 = x.borrow_mut();
        assert!(x.try_borrow().is_none());
    }

    #[test]
    fn no_imm_then_borrow_mut() {
        let x = RefCell::new(0);
        let _b1 = x.borrow();
        assert!(x.try_borrow_mut().is_none());
    }

    #[test]
    fn no_double_borrow_mut() {
        let x = RefCell::new(0);
        let _b1 = x.borrow_mut();
        assert!(x.try_borrow_mut().is_none());
    }

    #[test]
    fn imm_release_borrow_mut() {
        let x = RefCell::new(0);
        {
            let _b1 = x.borrow();
        }
        x.borrow_mut();
    }

    #[test]
    fn mut_release_borrow_mut() {
        let x = RefCell::new(0);
        {
            let _b1 = x.borrow_mut();
        }
        x.borrow();
    }

    #[test]
    fn double_borrow_single_release_no_borrow_mut() {
        let x = RefCell::new(0);
        let _b1 = x.borrow();
        {
            let _b2 = x.borrow();
        }
        assert!(x.try_borrow_mut().is_none());
    }

    #[test]
    #[should_fail]
    fn discard_doesnt_unborrow() {
        let x = RefCell::new(0);
        let _b = x.borrow();
        let _ = _b;
        let _b = x.borrow_mut();
    }
}
