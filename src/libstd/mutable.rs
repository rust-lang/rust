// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! A mutable memory location with dynamically checked borrow rules

use prelude::*;

use cast;
use util::NonCopyable;

/// A mutable memory location with dynamically checked borrow rules
#[no_freeze]
pub struct Mut<T> {
    priv value: T,
    priv borrow: BorrowFlag,
    priv nc: NonCopyable
}

// Values [1, MAX-1] represent the number of `Ref` active
// (will not outgrow its range since `uint` is the size of the address space)
type BorrowFlag = uint;
static UNUSED: BorrowFlag = 0;
static WRITING: BorrowFlag = -1;

impl<T> Mut<T> {
    /// Create a new `Mut` containing `value`
    pub fn new(value: T) -> Mut<T> {
        Mut {
            value: value,
            borrow: UNUSED,
            nc: NonCopyable
        }
    }

    /// Consumes the `Mut`, returning the wrapped value.
    pub fn unwrap(self) -> T {
        assert!(self.borrow == UNUSED);
        self.value
    }

    unsafe fn as_mut<'a>(&'a self) -> &'a mut Mut<T> {
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
            None => fail!("Mut<T> already mutably borrowed")
        }
    }

    /// Mutably borrows the wrapped value.
    ///
    /// The borrow lasts untile the returned `RefMut` exits scope. The value
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
    /// The borrow lasts untile the returned `RefMut` exits scope. The value
    /// cannot be borrowed while this borrow is active.
    ///
    /// # Failure
    ///
    /// Fails if the value is currently borrowed.
    pub fn borrow_mut<'a>(&'a self) -> RefMut<'a, T> {
        match self.try_borrow_mut() {
            Some(ptr) => ptr,
            None => fail!("Mut<T> already borrowed")
        }
    }

    /// Immutably borrows the wrapped value and applies `blk` to it.
    ///
    /// # Failure
    ///
    /// Fails if the value is currently mutably borrowed.
    #[inline]
    pub fn map<U>(&self, blk: |&T| -> U) -> U {
        let ptr = self.borrow();
        blk(ptr.get())
    }

    /// Mutably borrows the wrapped value and applies `blk` to it.
    ///
    /// # Failure
    ///
    /// Fails if the value is currently borrowed.
    #[inline]
    pub fn map_mut<U>(&self, blk: |&mut T| -> U) -> U {
        let mut ptr = self.borrow_mut();
        blk(ptr.get())
    }
}

impl<T: Clone> Clone for Mut<T> {
    fn clone(&self) -> Mut<T> {
        let x = self.borrow();
        Mut::new(x.get().clone())
    }
}

impl<T: DeepClone> DeepClone for Mut<T> {
    fn deep_clone(&self) -> Mut<T> {
        let x = self.borrow();
        Mut::new(x.get().deep_clone())
    }
}

impl<T: Eq> Eq for Mut<T> {
    fn eq(&self, other: &Mut<T>) -> bool {
        let a = self.borrow();
        let b = other.borrow();
        a.get() == b.get()
    }
}

/// Wraps a borrowed reference to a value in a `Mut` box.
pub struct Ref<'box, T> {
    priv parent: &'box Mut<T>
}

#[unsafe_destructor]
impl<'box, T> Drop for Ref<'box, T> {
    fn drop(&mut self) {
        assert!(self.parent.borrow != WRITING && self.parent.borrow != UNUSED);
        unsafe { self.parent.as_mut().borrow -= 1; }
    }
}

impl<'box, T> Ref<'box, T> {
    /// Retrieve an immutable reference to the stored value.
    #[inline]
    pub fn get<'a>(&'a self) -> &'a T {
        &self.parent.value
    }
}

/// Wraps a mutable borrowed reference to a value in a `Mut` box.
pub struct RefMut<'box, T> {
    priv parent: &'box mut Mut<T>
}

#[unsafe_destructor]
impl<'box, T> Drop for RefMut<'box, T> {
    fn drop(&mut self) {
        assert!(self.parent.borrow == WRITING);
        self.parent.borrow = UNUSED;
    }
}

impl<'box, T> RefMut<'box, T> {
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
    fn double_imm_borrow() {
        let x = Mut::new(0);
        let _b1 = x.borrow();
        x.borrow();
    }

    #[test]
    fn no_mut_then_imm_borrow() {
        let x = Mut::new(0);
        let _b1 = x.borrow_mut();
        assert!(x.try_borrow().is_none());
    }

    #[test]
    fn no_imm_then_borrow_mut() {
        let x = Mut::new(0);
        let _b1 = x.borrow();
        assert!(x.try_borrow_mut().is_none());
    }

    #[test]
    fn no_double_borrow_mut() {
        let x = Mut::new(0);
        let _b1 = x.borrow_mut();
        assert!(x.try_borrow_mut().is_none());
    }

    #[test]
    fn imm_release_borrow_mut() {
        let x = Mut::new(0);
        {
            let _b1 = x.borrow();
        }
        x.borrow_mut();
    }

    #[test]
    fn mut_release_borrow_mut() {
        let x = Mut::new(0);
        {
            let _b1 = x.borrow_mut();
        }
        x.borrow();
    }

    #[test]
    fn double_borrow_single_release_no_borrow_mut() {
        let x = Mut::new(0);
        let _b1 = x.borrow();
        {
            let _b2 = x.borrow();
        }
        assert!(x.try_borrow_mut().is_none());
    }

    #[test]
    fn map_ok() {
        let x = Mut::new(0);
        assert_eq!(1, x.map(|x| *x+1));
    }

    #[test]
    #[should_fail]
    fn mut_borrow_map() {
        let x = Mut::new(0);
        let _b1 = x.borrow_mut();
        x.map(|x| *x+1);
    }

    #[test]
    fn borrow_map() {
        let x = Mut::new(0);
        let _b1 = x.borrow();
        assert_eq!(1, x.map(|x| *x+1));
    }

    #[test]
    fn map_mut_ok() {
        let x = Mut::new(0);
        x.map_mut(|x| *x += 1);
        let b = x.borrow();
        assert_eq!(1, *b.get());
    }

    #[test]
    #[should_fail]
    fn borrow_map_mut() {
        let x = Mut::new(0);
        let _b = x.borrow();
        x.map_mut(|x| *x += 1);
    }
}
