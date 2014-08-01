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
//! use std::collections::HashMap;
//! use std::cell::RefCell;
//! use std::rc::Rc;
//!
//! fn main() {
//!     let shared_map: Rc<RefCell<_>> = Rc::new(RefCell::new(HashMap::new()));
//!     shared_map.borrow_mut().insert("africa", 92388i);
//!     shared_map.borrow_mut().insert("kyoto", 11837i);
//!     shared_map.borrow_mut().insert("piccadilly", 11826i);
//!     shared_map.borrow_mut().insert("marbles", 38i);
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
use cmp::PartialEq;
use kinds::{marker, Copy};
use ops::{Deref, DerefMut, Drop};
use option::{None, Option, Some};

/// A mutable memory location that admits only `Copy` data.
#[unstable = "likely to be renamed; otherwise stable"]
pub struct Cell<T> {
    value: UnsafeCell<T>,
    noshare: marker::NoShare,
}

#[stable]
impl<T:Copy> Cell<T> {
    /// Creates a new `Cell` containing the given value.
    pub fn new(value: T) -> Cell<T> {
        Cell {
            value: UnsafeCell::new(value),
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

#[unstable = "waiting for `Clone` trait to become stable"]
impl<T:Copy> Clone for Cell<T> {
    fn clone(&self) -> Cell<T> {
        Cell::new(self.get())
    }
}

#[unstable = "waiting for `PartialEq` trait to become stable"]
impl<T:PartialEq + Copy> PartialEq for Cell<T> {
    fn eq(&self, other: &Cell<T>) -> bool {
        self.get() == other.get()
    }
}

/// A mutable memory location with dynamically checked borrow rules
#[unstable = "likely to be renamed; otherwise stable"]
pub struct RefCell<T> {
    value: UnsafeCell<T>,
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
    #[stable]
    pub fn new(value: T) -> RefCell<T> {
        RefCell {
            value: UnsafeCell::new(value),
            borrow: Cell::new(UNUSED),
            nocopy: marker::NoCopy,
            noshare: marker::NoShare,
        }
    }

    /// Consumes the `RefCell`, returning the wrapped value.
    #[unstable = "may be renamed, depending on global conventions"]
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
    #[unstable = "may be renamed, depending on global conventions"]
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
    #[unstable]
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
    #[unstable = "may be renamed, depending on global conventions"]
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
    #[unstable]
    pub fn borrow_mut<'a>(&'a self) -> RefMut<'a, T> {
        match self.try_borrow_mut() {
            Some(ptr) => ptr,
            None => fail!("RefCell<T> already borrowed")
        }
    }
}

#[unstable = "waiting for `Clone` to become stable"]
impl<T: Clone> Clone for RefCell<T> {
    fn clone(&self) -> RefCell<T> {
        RefCell::new(self.borrow().clone())
    }
}

#[unstable = "waiting for `PartialEq` to become stable"]
impl<T: PartialEq> PartialEq for RefCell<T> {
    fn eq(&self, other: &RefCell<T>) -> bool {
        *self.borrow() == *other.borrow()
    }
}

/// Wraps a borrowed reference to a value in a `RefCell` box.
#[unstable]
pub struct Ref<'b, T> {
    // FIXME #12808: strange name to try to avoid interfering with
    // field accesses of the contained type via Deref
    _parent: &'b RefCell<T>
}

#[unsafe_destructor]
#[unstable]
impl<'b, T> Drop for Ref<'b, T> {
    fn drop(&mut self) {
        let borrow = self._parent.borrow.get();
        debug_assert!(borrow != WRITING && borrow != UNUSED);
        self._parent.borrow.set(borrow - 1);
    }
}

#[unstable = "waiting for `Deref` to become stable"]
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
#[experimental = "likely to be moved to a method, pending language changes"]
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
#[unstable]
pub struct RefMut<'b, T> {
    // FIXME #12808: strange name to try to avoid interfering with
    // field accesses of the contained type via Deref
    _parent: &'b RefCell<T>
}

#[unsafe_destructor]
#[unstable]
impl<'b, T> Drop for RefMut<'b, T> {
    fn drop(&mut self) {
        let borrow = self._parent.borrow.get();
        debug_assert!(borrow == WRITING);
        self._parent.borrow.set(UNUSED);
    }
}

#[unstable = "waiting for `Deref` to become stable"]
impl<'b, T> Deref<T> for RefMut<'b, T> {
    #[inline]
    fn deref<'a>(&'a self) -> &'a T {
        unsafe { &*self._parent.value.get() }
    }
}

#[unstable = "waiting for `DerefMut` to become stable"]
impl<'b, T> DerefMut<T> for RefMut<'b, T> {
    #[inline]
    fn deref_mut<'a>(&'a mut self) -> &'a mut T {
        unsafe { &mut *self._parent.value.get() }
    }
}

/// The core primitive for interior mutability in Rust.
///
/// `UnsafeCell` type that wraps a type T and indicates unsafe interior
/// operations on the wrapped type. Types with an `UnsafeCell<T>` field are
/// considered to have an *unsafe interior*. The `UnsafeCell` type is the only
/// legal way to obtain aliasable data that is considered mutable. In general,
/// transmuting an &T type into an &mut T is considered undefined behavior.
///
/// Although it is possible to put an `UnsafeCell<T>` into static item, it is
/// not permitted to take the address of the static item if the item is not
/// declared as mutable. This rule exists because immutable static items are
/// stored in read-only memory, and thus any attempt to mutate their interior
/// can cause segfaults. Immutable static items containing `UnsafeCell<T>`
/// instances are still useful as read-only initializers, however, so we do not
/// forbid them altogether.
///
/// Types like `Cell` and `RefCell` use this type to wrap their internal data.
///
/// `UnsafeCell` doesn't opt-out from any kind, instead, types with an
/// `UnsafeCell` interior are expected to opt-out from kinds themselves.
///
/// # Example:
///
/// ```rust
/// use std::cell::UnsafeCell;
/// use std::kinds::marker;
///
/// struct NotThreadSafe<T> {
///     value: UnsafeCell<T>,
///     marker: marker::NoShare
/// }
/// ```
///
/// **NOTE:** `UnsafeCell<T>` fields are public to allow static initializers. It
/// is not recommended to access its fields directly, `get` should be used
/// instead.
#[lang="unsafe"]
#[unstable = "this type may be renamed in the future"]
pub struct UnsafeCell<T> {
    /// Wrapped value
    ///
    /// This field should not be accessed directly, it is made public for static
    /// initializers.
    #[unstable]
    pub value: T,
}

impl<T> UnsafeCell<T> {
    /// Construct a new instance of `UnsafeCell` which will wrap the specified
    /// value.
    ///
    /// All access to the inner value through methods is `unsafe`, and it is
    /// highly discouraged to access the fields directly.
    #[stable]
    pub fn new(value: T) -> UnsafeCell<T> {
        UnsafeCell { value: value }
    }

    /// Gets a mutable pointer to the wrapped value.
    ///
    /// This function is unsafe as the pointer returned is an unsafe pointer and
    /// no guarantees are made about the aliasing of the pointers being handed
    /// out in this or other tasks.
    #[inline]
    #[unstable = "conventions around acquiring an inner reference are still \
                  under development"]
    pub unsafe fn get(&self) -> *mut T { &self.value as *const T as *mut T }

    /// Unwraps the value
    ///
    /// This function is unsafe because there is no guarantee that this or other
    /// tasks are currently inspecting the inner value.
    #[inline]
    #[unstable = "conventions around the name `unwrap` are still under \
                  development"]
    pub unsafe fn unwrap(self) -> T { self.value }
}
