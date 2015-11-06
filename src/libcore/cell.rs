// Copyright 2012-2014 The Rust Project Developers. See the COPYRIGHT
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
//! Values of the `Cell<T>` and `RefCell<T>` types may be mutated through shared references (i.e.
//! the common `&T` type), whereas most Rust types can only be mutated through unique (`&mut T`)
//! references. We say that `Cell<T>` and `RefCell<T>` provide 'interior mutability', in contrast
//! with typical Rust types that exhibit 'inherited mutability'.
//!
//! Cell types come in two flavors: `Cell<T>` and `RefCell<T>`. `Cell<T>` provides `get` and `set`
//! methods that change the interior value with a single method call. `Cell<T>` though is only
//! compatible with types that implement `Copy`. For other types, one must use the `RefCell<T>`
//! type, acquiring a write lock before mutating.
//!
//! `RefCell<T>` uses Rust's lifetimes to implement 'dynamic borrowing', a process whereby one can
//! claim temporary, exclusive, mutable access to the inner value. Borrows for `RefCell<T>`s are
//! tracked 'at runtime', unlike Rust's native reference types which are entirely tracked
//! statically, at compile time. Because `RefCell<T>` borrows are dynamic it is possible to attempt
//! to borrow a value that is already mutably borrowed; when this happens it results in thread
//! panic.
//!
//! # When to choose interior mutability
//!
//! The more common inherited mutability, where one must have unique access to mutate a value, is
//! one of the key language elements that enables Rust to reason strongly about pointer aliasing,
//! statically preventing crash bugs. Because of that, inherited mutability is preferred, and
//! interior mutability is something of a last resort. Since cell types enable mutation where it
//! would otherwise be disallowed though, there are occasions when interior mutability might be
//! appropriate, or even *must* be used, e.g.
//!
//! * Introducing mutability 'inside' of something immutable
//! * Implementation details of logically-immutable methods.
//! * Mutating implementations of `Clone`.
//!
//! ## Introducing mutability 'inside' of something immutable
//!
//! Many shared smart pointer types, including `Rc<T>` and `Arc<T>`, provide containers that can be
//! cloned and shared between multiple parties. Because the contained values may be
//! multiply-aliased, they can only be borrowed with `&`, not `&mut`. Without cells it would be
//! impossible to mutate data inside of these smart pointers at all.
//!
//! It's very common then to put a `RefCell<T>` inside shared pointer types to reintroduce
//! mutability:
//!
//! ```
//! use std::collections::HashMap;
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
//! Note that this example uses `Rc<T>` and not `Arc<T>`. `RefCell<T>`s are for single-threaded
//! scenarios. Consider using `RwLock<T>` or `Mutex<T>` if you need shared mutability in a
//! multi-threaded situation.
//!
//! ## Implementation details of logically-immutable methods
//!
//! Occasionally it may be desirable not to expose in an API that there is mutation happening
//! "under the hood". This may be because logically the operation is immutable, but e.g. caching
//! forces the implementation to perform mutation; or because you must employ mutation to implement
//! a trait method that was originally defined to take `&self`.
//!
//! ```
//! use std::cell::RefCell;
//!
//! struct Graph {
//!     edges: Vec<(i32, i32)>,
//!     span_tree_cache: RefCell<Option<Vec<(i32, i32)>>>
//! }
//!
//! impl Graph {
//!     fn minimum_spanning_tree(&self) -> Vec<(i32, i32)> {
//!         // Create a new scope to contain the lifetime of the
//!         // dynamic borrow
//!         {
//!             // Take a reference to the inside of cache cell
//!             let mut cache = self.span_tree_cache.borrow_mut();
//!             if cache.is_some() {
//!                 return cache.as_ref().unwrap().clone();
//!             }
//!
//!             let span_tree = self.calc_span_tree();
//!             *cache = Some(span_tree);
//!         }
//!
//!         // Recursive call to return the just-cached value.
//!         // Note that if we had not let the previous borrow
//!         // of the cache fall out of scope then the subsequent
//!         // recursive borrow would cause a dynamic thread panic.
//!         // This is the major hazard of using `RefCell`.
//!         self.minimum_spanning_tree()
//!     }
//! #   fn calc_span_tree(&self) -> Vec<(i32, i32)> { vec![] }
//! }
//! ```
//!
//! ## Mutating implementations of `Clone`
//!
//! This is simply a special - but common - case of the previous: hiding mutability for operations
//! that appear to be immutable. The `clone` method is expected to not change the source value, and
//! is declared to take `&self`, not `&mut self`. Therefore any mutation that happens in the
//! `clone` method must use cell types. For example, `Rc<T>` maintains its reference counts within a
//! `Cell<T>`.
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
//!     refcount: Cell<usize>
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

#![stable(feature = "rust1", since = "1.0.0")]

use clone::Clone;
use cmp::{PartialEq, Eq};
use default::Default;
use marker::{Copy, Send, Sync, Sized};
use ops::{Deref, DerefMut, Drop, FnOnce};
use option::Option;
use option::Option::{None, Some};

/// A mutable memory location that admits only `Copy` data.
///
/// See the [module-level documentation](index.html) for more.
#[stable(feature = "rust1", since = "1.0.0")]
pub struct Cell<T> {
    value: UnsafeCell<T>,
}

impl<T:Copy> Cell<T> {
    /// Creates a new `Cell` containing the given value.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::cell::Cell;
    ///
    /// let c = Cell::new(5);
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    #[inline]
    pub const fn new(value: T) -> Cell<T> {
        Cell {
            value: UnsafeCell::new(value),
        }
    }

    /// Returns a copy of the contained value.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::cell::Cell;
    ///
    /// let c = Cell::new(5);
    ///
    /// let five = c.get();
    /// ```
    #[inline]
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn get(&self) -> T {
        unsafe{ *self.value.get() }
    }

    /// Sets the contained value.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::cell::Cell;
    ///
    /// let c = Cell::new(5);
    ///
    /// c.set(10);
    /// ```
    #[inline]
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn set(&self, value: T) {
        unsafe {
            *self.value.get() = value;
        }
    }

    /// Returns a reference to the underlying `UnsafeCell`.
    ///
    /// # Safety
    ///
    /// This function is `unsafe` because `UnsafeCell`'s field is public.
    ///
    /// # Examples
    ///
    /// ```
    /// #![feature(as_unsafe_cell)]
    ///
    /// use std::cell::Cell;
    ///
    /// let c = Cell::new(5);
    ///
    /// let uc = unsafe { c.as_unsafe_cell() };
    /// ```
    #[inline]
    #[unstable(feature = "as_unsafe_cell", issue = "27708")]
    pub unsafe fn as_unsafe_cell(&self) -> &UnsafeCell<T> {
        &self.value
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
unsafe impl<T> Send for Cell<T> where T: Send {}

#[stable(feature = "rust1", since = "1.0.0")]
impl<T:Copy> Clone for Cell<T> {
    #[inline]
    fn clone(&self) -> Cell<T> {
        Cell::new(self.get())
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<T:Default + Copy> Default for Cell<T> {
    #[inline]
    fn default() -> Cell<T> {
        Cell::new(Default::default())
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<T:PartialEq + Copy> PartialEq for Cell<T> {
    #[inline]
    fn eq(&self, other: &Cell<T>) -> bool {
        self.get() == other.get()
    }
}

#[stable(feature = "cell_eq", since = "1.2.0")]
impl<T:Eq + Copy> Eq for Cell<T> {}

/// A mutable memory location with dynamically checked borrow rules
///
/// See the [module-level documentation](index.html) for more.
#[stable(feature = "rust1", since = "1.0.0")]
pub struct RefCell<T: ?Sized> {
    borrow: Cell<BorrowFlag>,
    value: UnsafeCell<T>,
}

/// An enumeration of values returned from the `state` method on a `RefCell<T>`.
#[derive(Copy, Clone, PartialEq, Eq, Debug)]
#[unstable(feature = "borrow_state", issue = "27733")]
pub enum BorrowState {
    /// The cell is currently being read, there is at least one active `borrow`.
    Reading,
    /// The cell is currently being written to, there is an active `borrow_mut`.
    Writing,
    /// There are no outstanding borrows on this cell.
    Unused,
}

// Values [1, MAX-1] represent the number of `Ref` active
// (will not outgrow its range since `usize` is the size of the address space)
type BorrowFlag = usize;
const UNUSED: BorrowFlag = 0;
const WRITING: BorrowFlag = !0;

impl<T> RefCell<T> {
    /// Creates a new `RefCell` containing `value`.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::cell::RefCell;
    ///
    /// let c = RefCell::new(5);
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    #[inline]
    pub const fn new(value: T) -> RefCell<T> {
        RefCell {
            value: UnsafeCell::new(value),
            borrow: Cell::new(UNUSED),
        }
    }

    /// Consumes the `RefCell`, returning the wrapped value.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::cell::RefCell;
    ///
    /// let c = RefCell::new(5);
    ///
    /// let five = c.into_inner();
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    #[inline]
    pub fn into_inner(self) -> T {
        // Since this function takes `self` (the `RefCell`) by value, the
        // compiler statically verifies that it is not currently borrowed.
        // Therefore the following assertion is just a `debug_assert!`.
        debug_assert!(self.borrow.get() == UNUSED);
        unsafe { self.value.into_inner() }
    }
}

impl<T: ?Sized> RefCell<T> {
    /// Query the current state of this `RefCell`
    ///
    /// The returned value can be dispatched on to determine if a call to
    /// `borrow` or `borrow_mut` would succeed.
    #[unstable(feature = "borrow_state", issue = "27733")]
    #[inline]
    pub fn borrow_state(&self) -> BorrowState {
        match self.borrow.get() {
            WRITING => BorrowState::Writing,
            UNUSED => BorrowState::Unused,
            _ => BorrowState::Reading,
        }
    }

    /// Immutably borrows the wrapped value.
    ///
    /// The borrow lasts until the returned `Ref` exits scope. Multiple
    /// immutable borrows can be taken out at the same time.
    ///
    /// # Panics
    ///
    /// Panics if the value is currently mutably borrowed.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::cell::RefCell;
    ///
    /// let c = RefCell::new(5);
    ///
    /// let borrowed_five = c.borrow();
    /// let borrowed_five2 = c.borrow();
    /// ```
    ///
    /// An example of panic:
    ///
    /// ```
    /// use std::cell::RefCell;
    /// use std::thread;
    ///
    /// let result = thread::spawn(move || {
    ///    let c = RefCell::new(5);
    ///    let m = c.borrow_mut();
    ///
    ///    let b = c.borrow(); // this causes a panic
    /// }).join();
    ///
    /// assert!(result.is_err());
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    #[inline]
    pub fn borrow(&self) -> Ref<T> {
        match BorrowRef::new(&self.borrow) {
            Some(b) => Ref {
                _value: unsafe { &*self.value.get() },
                _borrow: b,
            },
            None => panic!("RefCell<T> already mutably borrowed"),
        }
    }

    /// Mutably borrows the wrapped value.
    ///
    /// The borrow lasts until the returned `RefMut` exits scope. The value
    /// cannot be borrowed while this borrow is active.
    ///
    /// # Panics
    ///
    /// Panics if the value is currently borrowed.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::cell::RefCell;
    ///
    /// let c = RefCell::new(5);
    ///
    /// let borrowed_five = c.borrow_mut();
    /// ```
    ///
    /// An example of panic:
    ///
    /// ```
    /// use std::cell::RefCell;
    /// use std::thread;
    ///
    /// let result = thread::spawn(move || {
    ///    let c = RefCell::new(5);
    ///    let m = c.borrow();
    ///
    ///    let b = c.borrow_mut(); // this causes a panic
    /// }).join();
    ///
    /// assert!(result.is_err());
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    #[inline]
    pub fn borrow_mut(&self) -> RefMut<T> {
        match BorrowRefMut::new(&self.borrow) {
            Some(b) => RefMut {
                _value: unsafe { &mut *self.value.get() },
                _borrow: b,
            },
            None => panic!("RefCell<T> already borrowed"),
        }
    }

    /// Returns a reference to the underlying `UnsafeCell`.
    ///
    /// This can be used to circumvent `RefCell`'s safety checks.
    ///
    /// This function is `unsafe` because `UnsafeCell`'s field is public.
    #[inline]
    #[unstable(feature = "as_unsafe_cell", issue = "27708")]
    pub unsafe fn as_unsafe_cell(&self) -> &UnsafeCell<T> {
        &self.value
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
unsafe impl<T: ?Sized> Send for RefCell<T> where T: Send {}

#[stable(feature = "rust1", since = "1.0.0")]
impl<T: Clone> Clone for RefCell<T> {
    #[inline]
    fn clone(&self) -> RefCell<T> {
        RefCell::new(self.borrow().clone())
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<T:Default> Default for RefCell<T> {
    #[inline]
    fn default() -> RefCell<T> {
        RefCell::new(Default::default())
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<T: ?Sized + PartialEq> PartialEq for RefCell<T> {
    #[inline]
    fn eq(&self, other: &RefCell<T>) -> bool {
        *self.borrow() == *other.borrow()
    }
}

#[stable(feature = "cell_eq", since = "1.2.0")]
impl<T: ?Sized + Eq> Eq for RefCell<T> {}

struct BorrowRef<'b> {
    _borrow: &'b Cell<BorrowFlag>,
}

impl<'b> BorrowRef<'b> {
    #[inline]
    fn new(borrow: &'b Cell<BorrowFlag>) -> Option<BorrowRef<'b>> {
        match borrow.get() {
            WRITING => None,
            b => {
                borrow.set(b + 1);
                Some(BorrowRef { _borrow: borrow })
            },
        }
    }
}

impl<'b> Drop for BorrowRef<'b> {
    #[inline]
    fn drop(&mut self) {
        let borrow = self._borrow.get();
        debug_assert!(borrow != WRITING && borrow != UNUSED);
        self._borrow.set(borrow - 1);
    }
}

impl<'b> Clone for BorrowRef<'b> {
    #[inline]
    fn clone(&self) -> BorrowRef<'b> {
        // Since this Ref exists, we know the borrow flag
        // is not set to WRITING.
        let borrow = self._borrow.get();
        debug_assert!(borrow != WRITING && borrow != UNUSED);
        self._borrow.set(borrow + 1);
        BorrowRef { _borrow: self._borrow }
    }
}

/// Wraps a borrowed reference to a value in a `RefCell` box.
/// A wrapper type for an immutably borrowed value from a `RefCell<T>`.
///
/// See the [module-level documentation](index.html) for more.
#[stable(feature = "rust1", since = "1.0.0")]
pub struct Ref<'b, T: ?Sized + 'b> {
    // FIXME #12808: strange name to try to avoid interfering with
    // field accesses of the contained type via Deref
    _value: &'b T,
    _borrow: BorrowRef<'b>,
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<'b, T: ?Sized> Deref for Ref<'b, T> {
    type Target = T;

    #[inline]
    fn deref(&self) -> &T {
        self._value
    }
}

impl<'b, T: ?Sized> Ref<'b, T> {
    /// Copies a `Ref`.
    ///
    /// The `RefCell` is already immutably borrowed, so this cannot fail.
    ///
    /// This is an associated function that needs to be used as
    /// `Ref::clone(...)`.  A `Clone` implementation or a method would interfere
    /// with the widespread use of `r.borrow().clone()` to clone the contents of
    /// a `RefCell`.
    #[unstable(feature = "cell_extras",
               reason = "likely to be moved to a method, pending language changes",
               issue = "27746")]
    #[inline]
    pub fn clone(orig: &Ref<'b, T>) -> Ref<'b, T> {
        Ref {
            _value: orig._value,
            _borrow: orig._borrow.clone(),
        }
    }

    /// Make a new `Ref` for a component of the borrowed data.
    ///
    /// The `RefCell` is already immutably borrowed, so this cannot fail.
    ///
    /// This is an associated function that needs to be used as `Ref::map(...)`.
    /// A method would interfere with methods of the same name on the contents
    /// of a `RefCell` used through `Deref`.
    ///
    /// # Example
    ///
    /// ```
    /// #![feature(cell_extras)]
    ///
    /// use std::cell::{RefCell, Ref};
    ///
    /// let c = RefCell::new((5, 'b'));
    /// let b1: Ref<(u32, char)> = c.borrow();
    /// let b2: Ref<u32> = Ref::map(b1, |t| &t.0);
    /// assert_eq!(*b2, 5)
    /// ```
    #[unstable(feature = "cell_extras", reason = "recently added",
               issue = "27746")]
    #[inline]
    pub fn map<U: ?Sized, F>(orig: Ref<'b, T>, f: F) -> Ref<'b, U>
        where F: FnOnce(&T) -> &U
    {
        Ref {
            _value: f(orig._value),
            _borrow: orig._borrow,
        }
    }

    /// Make a new `Ref` for an optional component of the borrowed data, e.g. an
    /// enum variant.
    ///
    /// The `RefCell` is already immutably borrowed, so this cannot fail.
    ///
    /// This is an associated function that needs to be used as
    /// `Ref::filter_map(...)`.  A method would interfere with methods of the
    /// same name on the contents of a `RefCell` used through `Deref`.
    ///
    /// # Example
    ///
    /// ```
    /// # #![feature(cell_extras)]
    /// use std::cell::{RefCell, Ref};
    ///
    /// let c = RefCell::new(Ok(5));
    /// let b1: Ref<Result<u32, ()>> = c.borrow();
    /// let b2: Ref<u32> = Ref::filter_map(b1, |o| o.as_ref().ok()).unwrap();
    /// assert_eq!(*b2, 5)
    /// ```
    #[unstable(feature = "cell_extras", reason = "recently added",
               issue = "27746")]
    #[inline]
    pub fn filter_map<U: ?Sized, F>(orig: Ref<'b, T>, f: F) -> Option<Ref<'b, U>>
        where F: FnOnce(&T) -> Option<&U>
    {
        f(orig._value).map(move |new| Ref {
            _value: new,
            _borrow: orig._borrow,
        })
    }
}

impl<'b, T: ?Sized> RefMut<'b, T> {
    /// Make a new `RefMut` for a component of the borrowed data, e.g. an enum
    /// variant.
    ///
    /// The `RefCell` is already mutably borrowed, so this cannot fail.
    ///
    /// This is an associated function that needs to be used as
    /// `RefMut::map(...)`.  A method would interfere with methods of the same
    /// name on the contents of a `RefCell` used through `Deref`.
    ///
    /// # Example
    ///
    /// ```
    /// # #![feature(cell_extras)]
    /// use std::cell::{RefCell, RefMut};
    ///
    /// let c = RefCell::new((5, 'b'));
    /// {
    ///     let b1: RefMut<(u32, char)> = c.borrow_mut();
    ///     let mut b2: RefMut<u32> = RefMut::map(b1, |t| &mut t.0);
    ///     assert_eq!(*b2, 5);
    ///     *b2 = 42;
    /// }
    /// assert_eq!(*c.borrow(), (42, 'b'));
    /// ```
    #[unstable(feature = "cell_extras", reason = "recently added",
               issue = "27746")]
    #[inline]
    pub fn map<U: ?Sized, F>(orig: RefMut<'b, T>, f: F) -> RefMut<'b, U>
        where F: FnOnce(&mut T) -> &mut U
    {
        RefMut {
            _value: f(orig._value),
            _borrow: orig._borrow,
        }
    }

    /// Make a new `RefMut` for an optional component of the borrowed data, e.g.
    /// an enum variant.
    ///
    /// The `RefCell` is already mutably borrowed, so this cannot fail.
    ///
    /// This is an associated function that needs to be used as
    /// `RefMut::filter_map(...)`.  A method would interfere with methods of the
    /// same name on the contents of a `RefCell` used through `Deref`.
    ///
    /// # Example
    ///
    /// ```
    /// # #![feature(cell_extras)]
    /// use std::cell::{RefCell, RefMut};
    ///
    /// let c = RefCell::new(Ok(5));
    /// {
    ///     let b1: RefMut<Result<u32, ()>> = c.borrow_mut();
    ///     let mut b2: RefMut<u32> = RefMut::filter_map(b1, |o| {
    ///         o.as_mut().ok()
    ///     }).unwrap();
    ///     assert_eq!(*b2, 5);
    ///     *b2 = 42;
    /// }
    /// assert_eq!(*c.borrow(), Ok(42));
    /// ```
    #[unstable(feature = "cell_extras", reason = "recently added",
               issue = "27746")]
    #[inline]
    pub fn filter_map<U: ?Sized, F>(orig: RefMut<'b, T>, f: F) -> Option<RefMut<'b, U>>
        where F: FnOnce(&mut T) -> Option<&mut U>
    {
        let RefMut { _value, _borrow } = orig;
        f(_value).map(move |new| RefMut {
            _value: new,
            _borrow: _borrow,
        })
    }
}

struct BorrowRefMut<'b> {
    _borrow: &'b Cell<BorrowFlag>,
}

impl<'b> Drop for BorrowRefMut<'b> {
    #[inline]
    fn drop(&mut self) {
        let borrow = self._borrow.get();
        debug_assert!(borrow == WRITING);
        self._borrow.set(UNUSED);
    }
}

impl<'b> BorrowRefMut<'b> {
    #[inline]
    fn new(borrow: &'b Cell<BorrowFlag>) -> Option<BorrowRefMut<'b>> {
        match borrow.get() {
            UNUSED => {
                borrow.set(WRITING);
                Some(BorrowRefMut { _borrow: borrow })
            },
            _ => None,
        }
    }
}

/// A wrapper type for a mutably borrowed value from a `RefCell<T>`.
///
/// See the [module-level documentation](index.html) for more.
#[stable(feature = "rust1", since = "1.0.0")]
pub struct RefMut<'b, T: ?Sized + 'b> {
    // FIXME #12808: strange name to try to avoid interfering with
    // field accesses of the contained type via Deref
    _value: &'b mut T,
    _borrow: BorrowRefMut<'b>,
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<'b, T: ?Sized> Deref for RefMut<'b, T> {
    type Target = T;

    #[inline]
    fn deref(&self) -> &T {
        self._value
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<'b, T: ?Sized> DerefMut for RefMut<'b, T> {
    #[inline]
    fn deref_mut(&mut self) -> &mut T {
        self._value
    }
}

/// The core primitive for interior mutability in Rust.
///
/// `UnsafeCell<T>` is a type that wraps some `T` and indicates unsafe interior operations on the
/// wrapped type. Types with an `UnsafeCell<T>` field are considered to have an 'unsafe interior'.
/// The `UnsafeCell<T>` type is the only legal way to obtain aliasable data that is considered
/// mutable. In general, transmuting an `&T` type into an `&mut T` is considered undefined behavior.
///
/// Types like `Cell<T>` and `RefCell<T>` use this type to wrap their internal data.
///
/// # Examples
///
/// ```
/// use std::cell::UnsafeCell;
/// use std::marker::Sync;
///
/// struct NotThreadSafe<T> {
///     value: UnsafeCell<T>,
/// }
///
/// unsafe impl<T> Sync for NotThreadSafe<T> {}
/// ```
#[lang = "unsafe_cell"]
#[stable(feature = "rust1", since = "1.0.0")]
pub struct UnsafeCell<T: ?Sized> {
    value: T,
}

impl<T: ?Sized> !Sync for UnsafeCell<T> {}

impl<T> UnsafeCell<T> {
    /// Constructs a new instance of `UnsafeCell` which will wrap the specified
    /// value.
    ///
    /// All access to the inner value through methods is `unsafe`.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::cell::UnsafeCell;
    ///
    /// let uc = UnsafeCell::new(5);
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    #[inline]
    pub const fn new(value: T) -> UnsafeCell<T> {
        UnsafeCell { value: value }
    }

    /// Unwraps the value.
    ///
    /// # Safety
    ///
    /// This function is unsafe because this thread or another thread may currently be
    /// inspecting the inner value.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::cell::UnsafeCell;
    ///
    /// let uc = UnsafeCell::new(5);
    ///
    /// let five = unsafe { uc.into_inner() };
    /// ```
    #[inline]
    #[stable(feature = "rust1", since = "1.0.0")]
    pub unsafe fn into_inner(self) -> T {
        self.value
    }
}

impl<T: ?Sized> UnsafeCell<T> {
    /// Gets a mutable pointer to the wrapped value.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::cell::UnsafeCell;
    ///
    /// let uc = UnsafeCell::new(5);
    ///
    /// let five = uc.get();
    /// ```
    #[inline]
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn get(&self) -> *mut T {
        &self.value as *const T as *mut T
    }
}
