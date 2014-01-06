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

use prelude::*;
use cast;
use util::NonCopyable;

/// A mutable memory location that admits only `Pod` data.
#[no_freeze]
#[deriving(Clone)]
pub struct Cell<T> {
    priv value: T,
}

impl<T: ::kinds::Pod> Cell<T> {
    /// Creates a new `Cell` containing the given value.
    pub fn new(value: T) -> Cell<T> {
        Cell {
            value: value,
        }
    }

    /// Returns a copy of the contained value.
    #[inline]
    pub fn get(&self) -> T {
        self.value
    }

    /// Sets the contained value.
    #[inline]
    pub fn set(&self, value: T) {
        unsafe {
            *cast::transmute_mut(&self.value) = value
        }
    }
}

/// A mutable memory location with dynamically checked borrow rules
#[no_freeze]
pub struct RefCell<T> {
    priv value: T,
    priv borrow: BorrowFlag,
    priv nc: NonCopyable
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
            value: value,
            borrow: UNUSED,
            nc: NonCopyable
        }
    }

    /// Consumes the `RefCell`, returning the wrapped value.
    pub fn unwrap(self) -> T {
        assert!(self.borrow == UNUSED);
        self.value
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

    /// Immutably borrows the wrapped value and applies `blk` to it.
    ///
    /// # Failure
    ///
    /// Fails if the value is currently mutably borrowed.
    #[inline]
    pub fn with<U>(&self, blk: |&T| -> U) -> U {
        let ptr = self.borrow();
        blk(ptr.get())
    }

    /// Mutably borrows the wrapped value and applies `blk` to it.
    ///
    /// # Failure
    ///
    /// Fails if the value is currently borrowed.
    #[inline]
    pub fn with_mut<U>(&self, blk: |&mut T| -> U) -> U {
        let mut ptr = self.borrow_mut();
        blk(ptr.get())
    }

    /// Sets the value, replacing what was there.
    ///
    /// # Failure
    ///
    /// Fails if the value is currently borrowed.
    #[inline]
    pub fn set(&self, value: T) {
        let mut reference = self.borrow_mut();
        *reference.get() = value
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
        let reference = self.borrow();
        (*reference.get()).clone()
    }
}

impl<T: Clone> Clone for RefCell<T> {
    fn clone(&self) -> RefCell<T> {
        let x = self.borrow();
        RefCell::new(x.get().clone())
    }
}

impl<T: DeepClone> DeepClone for RefCell<T> {
    fn deep_clone(&self) -> RefCell<T> {
        let x = self.borrow();
        RefCell::new(x.get().deep_clone())
    }
}

impl<T: Eq> Eq for RefCell<T> {
    fn eq(&self, other: &RefCell<T>) -> bool {
        let a = self.borrow();
        let b = other.borrow();
        a.get() == b.get()
    }
}

/// Wraps a borrowed reference to a value in a `RefCell` box.
pub struct Ref<'b, T> {
    priv parent: &'b RefCell<T>
}

#[unsafe_destructor]
impl<'b, T> Drop for Ref<'b, T> {
    fn drop(&mut self) {
        assert!(self.parent.borrow != WRITING && self.parent.borrow != UNUSED);
        unsafe { self.parent.as_mut().borrow -= 1; }
    }
}

impl<'b, T> Ref<'b, T> {
    /// Retrieve an immutable reference to the stored value.
    #[inline]
    pub fn get<'a>(&'a self) -> &'a T {
        &self.parent.value
    }
}

/// Wraps a mutable borrowed reference to a value in a `RefCell` box.
pub struct RefMut<'b, T> {
    priv parent: &'b mut RefCell<T>
}

#[unsafe_destructor]
impl<'b, T> Drop for RefMut<'b, T> {
    fn drop(&mut self) {
        assert!(self.parent.borrow == WRITING);
        self.parent.borrow = UNUSED;
    }
}

impl<'b, T> RefMut<'b, T> {
    /// Retrieve a mutable reference to the stored value.
    #[inline]
    pub fn get<'a>(&'a mut self) -> &'a mut T {
        &mut self.parent.value
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn smoketest_cell() {
        let x = Cell::new(10);
        assert_eq!(x.get(), 10);
        x.set(20);
        assert_eq!(x.get(), 20);

        let y = Cell::new((30, 40));
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
    fn with_ok() {
        let x = RefCell::new(0);
        assert_eq!(1, x.with(|x| *x+1));
    }

    #[test]
    #[should_fail]
    fn mut_borrow_with() {
        let x = RefCell::new(0);
        let _b1 = x.borrow_mut();
        x.with(|x| *x+1);
    }

    #[test]
    fn borrow_with() {
        let x = RefCell::new(0);
        let _b1 = x.borrow();
        assert_eq!(1, x.with(|x| *x+1));
    }

    #[test]
    fn with_mut_ok() {
        let x = RefCell::new(0);
        x.with_mut(|x| *x += 1);
        let b = x.borrow();
        assert_eq!(1, *b.get());
    }

    #[test]
    #[should_fail]
    fn borrow_with_mut() {
        let x = RefCell::new(0);
        let _b = x.borrow();
        x.with_mut(|x| *x += 1);
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
