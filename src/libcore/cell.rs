// Copyright 2012-2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Shareable mutable containers.
//!
//! Values of the `Cell` and `RefCell` types may be mutated through
//! shared references (i.e. the common `&T` type), whereas most Rust
//! types can only be mutated through unique (`&mut T`) references. We
//! say that `Cell` and `RefCell` provide *interior mutability*, in
//! contrast with typical Rust types that exhibit *inherited
//! mutability*.
//!
//! Cell types come in two flavors: `Cell` and `RefCell`. `Cell`
//! provides `get` and `set` methods that change the
//! interior value with a single method call. `Cell` though is only
//! compatible with types that implement `Copy`. For other types,
//! one must use the `RefCell` type, acquiring a write lock before
//! mutating.
//!
//! `RefCell` uses Rust's lifetimes to implement *dynamic borrowing*,
//! a process whereby one can claim temporary, exclusive, mutable
//! access to the inner value. Borrows for `RefCell`s are tracked *at
//! runtime*, unlike Rust's native reference types which are entirely
//! tracked statically, at compile time. Because `RefCell` borrows are
//! dynamic it is possible to attempt to borrow a value that is
//! already mutably borrowed; when this happens it results in task
//! failure.
//!
//! # When to choose interior mutability
//!
//! The more common inherited mutability, where one must have unique
//! access to mutate a value, is one of the key language elements that
//! enables Rust to reason strongly about pointer aliasing, statically
//! preventing crash bugs. Because of that, inherited mutability is
//! preferred, and interior mutability is something of a last
//! resort. Since cell types enable mutation where it would otherwise
//! be disallowed though, there are occasions when interior
//! mutability might be appropriate, or even *must* be used, e.g.
//!
//! * Introducing inherited mutability roots to shared types.
//! * Implementation details of logically-immutable methods.
//! * Mutating implementations of `clone`.
//!
//! ## Introducing inherited mutability roots to shared types
//!
//! Shared smart pointer types, including `Rc` and `Arc`, provide
//! containers that can be cloned and shared between multiple parties.
//! Because the contained values may be multiply-aliased, they can
//! only be borrowed as shared references, not mutable references.
//! Without cells it would be impossible to mutate data inside of
//! shared boxes at all!
//!
//! It's very common then to put a `RefCell` inside shared pointer
//! types to reintroduce mutability:
//!
//! ```
//! extern crate collections;
//!
//! use collections::HashMap;
//! use std::cell::RefCell;
//! use std::rc::Rc;
//!
//! fn main() {
//!     let shared_map: Rc<RefCell<_>> = Rc::new(RefCell::new(HashMap::new()));
//!     shared_map.borrow_mut().insert("africa", 92388);
//!     shared_map.borrow_mut().insert("kyoto", 11837);
//!     shared_map.borrow_mut().insert("piccadilly", 11826);
//!     shared_map.borrow_mut().insert("marbles", 38);
//! }
//! ```
//!
//! ## Implementation details of logically-immutable methods
//!
//! Occasionally it may be desirable not to expose in an API that
//! there is mutation happening "under the hood". This may be because
//! logically the operation is immutable, but e.g. caching forces the
//! implementation to perform mutation; or because you must employ
//! mutation to implement a trait method that was originally defined
//! to take `&self`.
//!
//! ```
//! extern crate collections;
//!
//! use std::cell::RefCell;
//!
//! struct Graph {
//!     edges: Vec<(uint, uint)>,
//!     span_tree_cache: RefCell<Option<Vec<(uint, uint)>>>
//! }
//!
//! impl Graph {
//!     fn minimum_spanning_tree(&self) -> Vec<(uint, uint)> {
//!         // Create a new scope to contain the lifetime of the
//!         // dynamic borrow
//!         {
//!             // Take a reference to the inside of cache cell
//!             let mut cache = self.span_tree_cache.borrow_mut();
//!             if cache.is_some() {
//!                 return cache.get_ref().clone();
//!             }
//!
//!             let span_tree = self.calc_span_tree();
//!             *cache = Some(span_tree);
//!         }
//!
//!         // Recursive call to return the just-cached value.
//!         // Note that if we had not let the previous borrow
//!         // of the cache fall out of scope then the subsequent
//!         // recursive borrow would cause a dynamic task failure.
//!         // This is the major hazard of using `RefCell`.
//!         self.minimum_spanning_tree()
//!     }
//! #   fn calc_span_tree(&self) -> Vec<(uint, uint)> { vec![] }
//! }
//! # fn main() { }
//! ```
//!
//! ## Mutating implementations of `clone`
//!
//! This is simply a special - but common - case of the previous:
//! hiding mutability for operations that appear to be immutable.
//! The `clone` method is expected to not change the source value, and
//! is declared to take `&self`, not `&mut self`. Therefore any
//! mutation that happens in the `clone` method must use cell
//! types. For example, `Rc` maintains its reference counts within a
//! `Cell`.
//!
//! ```
//! use std::cell::Cell;
//!
//! struct Rc<T> {
//!     ptr: *mut RcBox<T>
//! }
//!
//! struct RcBox<T> {
//!     value: T,
//!     refcount: Cell<uint>
//! }
//!
//! impl<T> Clone for Rc<T> {
//!     fn clone(&self) -> Rc<T> {
//!         unsafe {
//!             (*self.ptr).refcount.set((*self.ptr).refcount.get() + 1);
//!             Rc { ptr: self.ptr }
//!         }
//!     }
//! }
//! ```
//!
// FIXME: Explain difference between Cell and RefCell
// FIXME: Downsides to interior mutability
// FIXME: Can't be shared between threads. Dynamic borrows
// FIXME: Relationship to Atomic types and RWLock

use clone::Clone;
use cmp::Eq;
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

/// A mutable memory location with dynamically checked borrow rules
pub struct RefCell<T> {
    value: Unsafe<T>,
    borrow: Cell<BorrowFlag>,
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
            borrow: Cell::new(UNUSED),
            nocopy: marker::NoCopy,
            noshare: marker::NoShare,
        }
    }

    /// Consumes the `RefCell`, returning the wrapped value.
    pub fn unwrap(self) -> T {
        debug_assert!(self.borrow.get() == UNUSED);
        unsafe{self.value.unwrap()}
    }

    /// Attempts to immutably borrow the wrapped value.
    ///
    /// The borrow lasts until the returned `Ref` exits scope. Multiple
    /// immutable borrows can be taken out at the same time.
    ///
    /// Returns `None` if the value is currently mutably borrowed.
    pub fn try_borrow<'a>(&'a self) -> Option<Ref<'a, T>> {
        match self.borrow.get() {
            WRITING => None,
            borrow => {
                self.borrow.set(borrow + 1);
                Some(Ref { _parent: self })
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
        match self.borrow.get() {
            UNUSED => {
                self.borrow.set(WRITING);
                Some(RefMut { _parent: self })
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
}

impl<T: Clone> Clone for RefCell<T> {
    fn clone(&self) -> RefCell<T> {
        RefCell::new(self.borrow().clone())
    }
}

impl<T: Eq> Eq for RefCell<T> {
    fn eq(&self, other: &RefCell<T>) -> bool {
        *self.borrow() == *other.borrow()
    }
}

/// Wraps a borrowed reference to a value in a `RefCell` box.
pub struct Ref<'b, T> {
    // FIXME #12808: strange name to try to avoid interfering with
    // field accesses of the contained type via Deref
    _parent: &'b RefCell<T>
}

#[unsafe_destructor]
impl<'b, T> Drop for Ref<'b, T> {
    fn drop(&mut self) {
        let borrow = self._parent.borrow.get();
        debug_assert!(borrow != WRITING && borrow != UNUSED);
        self._parent.borrow.set(borrow - 1);
    }
}

impl<'b, T> Deref<T> for Ref<'b, T> {
    #[inline]
    fn deref<'a>(&'a self) -> &'a T {
        unsafe { &*self._parent.value.get() }
    }
}

/// Copy a `Ref`.
///
/// The `RefCell` is already immutably borrowed, so this cannot fail.
///
/// A `Clone` implementation would interfere with the widespread
/// use of `r.borrow().clone()` to clone the contents of a `RefCell`.
#[experimental]
pub fn clone_ref<'b, T>(orig: &Ref<'b, T>) -> Ref<'b, T> {
    // Since this Ref exists, we know the borrow flag
    // is not set to WRITING.
    let borrow = orig._parent.borrow.get();
    debug_assert!(borrow != WRITING && borrow != UNUSED);
    orig._parent.borrow.set(borrow + 1);

    Ref {
        _parent: orig._parent,
    }
}

/// Wraps a mutable borrowed reference to a value in a `RefCell` box.
pub struct RefMut<'b, T> {
    // FIXME #12808: strange name to try to avoid interfering with
    // field accesses of the contained type via Deref
    _parent: &'b RefCell<T>
}

#[unsafe_destructor]
impl<'b, T> Drop for RefMut<'b, T> {
    fn drop(&mut self) {
        let borrow = self._parent.borrow.get();
        debug_assert!(borrow == WRITING);
        self._parent.borrow.set(UNUSED);
    }
}

impl<'b, T> Deref<T> for RefMut<'b, T> {
    #[inline]
    fn deref<'a>(&'a self) -> &'a T {
        unsafe { &*self._parent.value.get() }
    }
}

impl<'b, T> DerefMut<T> for RefMut<'b, T> {
    #[inline]
    fn deref_mut<'a>(&'a mut self) -> &'a mut T {
        unsafe { &mut *self._parent.value.get() }
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn smoketest_cell() {
        let x = Cell::new(10);
        assert!(x == Cell::new(10));
        assert!(x.get() == 10);
        x.set(20);
        assert!(x == Cell::new(20));
        assert!(x.get() == 20);

        let y = Cell::new((30, 40));
        assert!(y == Cell::new((30, 40)));
        assert!(y.get() == (30, 40));
    }

    #[test]
    fn cell_has_sensible_show() {
        use str::StrSlice;
        use realstd::str::Str;

        let x = Cell::new("foo bar");
        assert!(format!("{}", x).as_slice().contains(x.get()));

        x.set("baz qux");
        assert!(format!("{}", x).as_slice().contains(x.get()));
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

    #[test]
    #[allow(experimental)]
    fn clone_ref_updates_flag() {
        let x = RefCell::new(0);
        {
            let b1 = x.borrow();
            assert!(x.try_borrow_mut().is_none());
            {
                let _b2 = clone_ref(&b1);
                assert!(x.try_borrow_mut().is_none());
            }
            assert!(x.try_borrow_mut().is_none());
        }
        assert!(x.try_borrow_mut().is_some());
    }
}
