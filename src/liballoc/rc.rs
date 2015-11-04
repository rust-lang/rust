// Copyright 2013-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![allow(deprecated)]

//! Thread-local reference-counted boxes (the `Rc<T>` type).
//!
//! The `Rc<T>` type provides shared ownership of an immutable value.
//! Destruction is deterministic, and will occur as soon as the last owner is
//! gone. It is marked as non-sendable because it avoids the overhead of atomic
//! reference counting.
//!
//! The `downgrade` method can be used to create a non-owning `Weak<T>` pointer
//! to the box. A `Weak<T>` pointer can be upgraded to an `Rc<T>` pointer, but
//! will return `None` if the value has already been dropped.
//!
//! For example, a tree with parent pointers can be represented by putting the
//! nodes behind strong `Rc<T>` pointers, and then storing the parent pointers
//! as `Weak<T>` pointers.
//!
//! # Examples
//!
//! Consider a scenario where a set of `Gadget`s are owned by a given `Owner`.
//! We want to have our `Gadget`s point to their `Owner`. We can't do this with
//! unique ownership, because more than one gadget may belong to the same
//! `Owner`. `Rc<T>` allows us to share an `Owner` between multiple `Gadget`s,
//! and have the `Owner` remain allocated as long as any `Gadget` points at it.
//!
//! ```rust
//! use std::rc::Rc;
//!
//! struct Owner {
//!     name: String
//!     // ...other fields
//! }
//!
//! struct Gadget {
//!     id: i32,
//!     owner: Rc<Owner>
//!     // ...other fields
//! }
//!
//! fn main() {
//!     // Create a reference counted Owner.
//!     let gadget_owner : Rc<Owner> = Rc::new(
//!         Owner { name: String::from("Gadget Man") }
//!     );
//!
//!     // Create Gadgets belonging to gadget_owner. To increment the reference
//!     // count we clone the `Rc<T>` object.
//!     let gadget1 = Gadget { id: 1, owner: gadget_owner.clone() };
//!     let gadget2 = Gadget { id: 2, owner: gadget_owner.clone() };
//!
//!     drop(gadget_owner);
//!
//!     // Despite dropping gadget_owner, we're still able to print out the name
//!     // of the Owner of the Gadgets. This is because we've only dropped the
//!     // reference count object, not the Owner it wraps. As long as there are
//!     // other `Rc<T>` objects pointing at the same Owner, it will remain
//!     // allocated. Notice that the `Rc<T>` wrapper around Gadget.owner gets
//!     // automatically dereferenced for us.
//!     println!("Gadget {} owned by {}", gadget1.id, gadget1.owner.name);
//!     println!("Gadget {} owned by {}", gadget2.id, gadget2.owner.name);
//!
//!     // At the end of the method, gadget1 and gadget2 get destroyed, and with
//!     // them the last counted references to our Owner. Gadget Man now gets
//!     // destroyed as well.
//! }
//! ```
//!
//! If our requirements change, and we also need to be able to traverse from
//! Owner → Gadget, we will run into problems: an `Rc<T>` pointer from Owner
//! → Gadget introduces a cycle between the objects. This means that their
//! reference counts can never reach 0, and the objects will remain allocated: a
//! memory leak. In order to get around this, we can use `Weak<T>` pointers.
//! These pointers don't contribute to the total count.
//!
//! Rust actually makes it somewhat difficult to produce this loop in the first
//! place: in order to end up with two objects that point at each other, one of
//! them needs to be mutable. This is problematic because `Rc<T>` enforces
//! memory safety by only giving out shared references to the object it wraps,
//! and these don't allow direct mutation. We need to wrap the part of the
//! object we wish to mutate in a `RefCell`, which provides *interior
//! mutability*: a method to achieve mutability through a shared reference.
//! `RefCell` enforces Rust's borrowing rules at runtime.  Read the `Cell`
//! documentation for more details on interior mutability.
//!
//! ```rust
//! use std::rc::Rc;
//! use std::rc::Weak;
//! use std::cell::RefCell;
//!
//! struct Owner {
//!     name: String,
//!     gadgets: RefCell<Vec<Weak<Gadget>>>,
//!     // ...other fields
//! }
//!
//! struct Gadget {
//!     id: i32,
//!     owner: Rc<Owner>,
//!     // ...other fields
//! }
//!
//! fn main() {
//!     // Create a reference counted Owner. Note the fact that we've put the
//!     // Owner's vector of Gadgets inside a RefCell so that we can mutate it
//!     // through a shared reference.
//!     let gadget_owner : Rc<Owner> = Rc::new(
//!         Owner {
//!             name: "Gadget Man".to_string(),
//!             gadgets: RefCell::new(Vec::new()),
//!         }
//!     );
//!
//!     // Create Gadgets belonging to gadget_owner as before.
//!     let gadget1 = Rc::new(Gadget{id: 1, owner: gadget_owner.clone()});
//!     let gadget2 = Rc::new(Gadget{id: 2, owner: gadget_owner.clone()});
//!
//!     // Add the Gadgets to their Owner. To do this we mutably borrow from
//!     // the RefCell holding the Owner's Gadgets.
//!     gadget_owner.gadgets.borrow_mut().push(Rc::downgrade(&gadget1));
//!     gadget_owner.gadgets.borrow_mut().push(Rc::downgrade(&gadget2));
//!
//!     // Iterate over our Gadgets, printing their details out
//!     for gadget_opt in gadget_owner.gadgets.borrow().iter() {
//!
//!         // gadget_opt is a Weak<Gadget>. Since weak pointers can't guarantee
//!         // that their object is still allocated, we need to call upgrade()
//!         // on them to turn them into a strong reference. This returns an
//!         // Option, which contains a reference to our object if it still
//!         // exists.
//!         let gadget = gadget_opt.upgrade().unwrap();
//!         println!("Gadget {} owned by {}", gadget.id, gadget.owner.name);
//!     }
//!
//!     // At the end of the method, gadget_owner, gadget1 and gadget2 get
//!     // destroyed. There are now no strong (`Rc<T>`) references to the gadgets.
//!     // Once they get destroyed, the Gadgets get destroyed. This zeroes the
//!     // reference count on Gadget Man, they get destroyed as well.
//! }
//! ```

#![stable(feature = "rust1", since = "1.0.0")]

#[cfg(not(test))]
use boxed::Box;
#[cfg(test)]
use std::boxed::Box;

use core::borrow;
use core::cell::Cell;
use core::cmp::Ordering;
use core::fmt;
use core::hash::{Hasher, Hash};
use core::intrinsics::{assume, abort};
use core::marker::{self, Unsize};
use core::mem::{self, align_of_val, size_of_val, forget};
use core::ops::{CoerceUnsized, Deref};
use core::ptr::{self, Shared};
use core::convert::From;

use heap::deallocate;

struct RcBox<T: ?Sized> {
    strong: Cell<usize>,
    weak: Cell<usize>,
    value: T,
}


/// A reference-counted pointer type over an immutable value.
///
/// See the [module level documentation](./index.html) for more details.
#[unsafe_no_drop_flag]
#[stable(feature = "rust1", since = "1.0.0")]
pub struct Rc<T: ?Sized> {
    // FIXME #12808: strange names to try to avoid interfering with field
    // accesses of the contained type via Deref
    _ptr: Shared<RcBox<T>>,
}

impl<T: ?Sized> !marker::Send for Rc<T> {}
impl<T: ?Sized> !marker::Sync for Rc<T> {}

#[cfg(not(stage0))] // remove cfg after new snapshot
impl<T: ?Sized+Unsize<U>, U: ?Sized> CoerceUnsized<Rc<U>> for Rc<T> {}

impl<T> Rc<T> {
    /// Constructs a new `Rc<T>`.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::rc::Rc;
    ///
    /// let five = Rc::new(5);
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn new(value: T) -> Rc<T> {
        unsafe {
            Rc {
                // there is an implicit weak pointer owned by all the strong
                // pointers, which ensures that the weak destructor never frees
                // the allocation while the strong destructor is running, even
                // if the weak pointer is stored inside the strong one.
                _ptr: Shared::new(Box::into_raw(box RcBox {
                    strong: Cell::new(1),
                    weak: Cell::new(1),
                    value: value,
                })),
            }
        }
    }

    /// Unwraps the contained value if the `Rc<T>` has only one strong reference.
    /// This will succeed even if there are outstanding weak references.
    ///
    /// Otherwise, an `Err` is returned with the same `Rc<T>`.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::rc::Rc;
    ///
    /// let x = Rc::new(3);
    /// assert_eq!(Rc::try_unwrap(x), Ok(3));
    ///
    /// let x = Rc::new(4);
    /// let _y = x.clone();
    /// assert_eq!(Rc::try_unwrap(x), Err(Rc::new(4)));
    /// ```
    #[inline]
    #[stable(feature = "rc_unique", since = "1.4.0")]
    pub fn try_unwrap(this: Self) -> Result<T, Self> {
        if Rc::would_unwrap(&this) {
            unsafe {
                let val = ptr::read(&*this); // copy the contained object

                // Indicate to Weaks that they can't be promoted by decrememting
                // the strong count, and then remove the implicit "strong weak"
                // pointer while also handling drop logic by just crafting a
                // fake Weak.
                this.dec_strong();
                let _weak = Weak { _ptr: this._ptr };
                forget(this);
                Ok(val)
            }
        } else {
            Err(this)
        }
    }

    /// Checks if `Rc::try_unwrap` would return `Ok`.
    #[unstable(feature = "rc_would_unwrap",
               reason = "just added for niche usecase",
               issue = "28356")]
    pub fn would_unwrap(this: &Self) -> bool {
        Rc::strong_count(&this) == 1
    }
}

impl<T: ?Sized> Rc<T> {
    /// Downgrades the `Rc<T>` to a `Weak<T>` reference.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::rc::Rc;
    ///
    /// let five = Rc::new(5);
    ///
    /// let weak_five = Rc::downgrade(&five);
    /// ```
    #[stable(feature = "rc_weak", since = "1.4.0")]
    pub fn downgrade(this: &Self) -> Weak<T> {
        this.inc_weak();
        Weak { _ptr: this._ptr }
    }

    /// Get the number of weak references to this value.
    #[inline]
    #[unstable(feature = "rc_counts", reason = "not clearly useful",
               issue = "28356")]
    pub fn weak_count(this: &Self) -> usize {
        this.weak() - 1
    }

    /// Get the number of strong references to this value.
    #[inline]
    #[unstable(feature = "rc_counts", reason = "not clearly useful",
               issue = "28356")]
    pub fn strong_count(this: &Self) -> usize {
        this.strong()
    }

    /// Returns true if there are no other `Rc` or `Weak<T>` values that share
    /// the same inner value.
    ///
    /// # Examples
    ///
    /// ```
    /// #![feature(rc_counts)]
    ///
    /// use std::rc::Rc;
    ///
    /// let five = Rc::new(5);
    ///
    /// assert!(Rc::is_unique(&five));
    /// ```
    #[inline]
    #[unstable(feature = "rc_counts", reason = "uniqueness has unclear meaning",
               issue = "28356")]
    pub fn is_unique(this: &Self) -> bool {
        Rc::weak_count(this) == 0 && Rc::strong_count(this) == 1
    }

    /// Returns a mutable reference to the contained value if the `Rc<T>` has
    /// one strong reference and no weak references.
    ///
    /// Returns `None` if the `Rc<T>` is not unique.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::rc::Rc;
    ///
    /// let mut x = Rc::new(3);
    /// *Rc::get_mut(&mut x).unwrap() = 4;
    /// assert_eq!(*x, 4);
    ///
    /// let _y = x.clone();
    /// assert!(Rc::get_mut(&mut x).is_none());
    /// ```
    #[inline]
    #[stable(feature = "rc_unique", since = "1.4.0")]
    pub fn get_mut(this: &mut Self) -> Option<&mut T> {
        if Rc::is_unique(this) {
            let inner = unsafe { &mut **this._ptr };
            Some(&mut inner.value)
        } else {
            None
        }
    }
}

impl<T: Clone> Rc<T> {
    #[inline]
    #[unstable(feature = "rc_make_unique", reason = "renamed to Rc::make_mut",
               issue = "27718")]
    #[deprecated(since = "1.4.0", reason = "renamed to Rc::make_mut")]
    pub fn make_unique(&mut self) -> &mut T {
        Rc::make_mut(self)
    }

    /// Make a mutable reference into the given `Rc<T>` by cloning the inner
    /// data if the `Rc<T>` doesn't have one strong reference and no weak
    /// references.
    ///
    /// This is also referred to as a copy-on-write.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::rc::Rc;
    ///
    /// let mut data = Rc::new(5);
    ///
    /// *Rc::make_mut(&mut data) += 1;             // Won't clone anything
    /// let mut other_data = data.clone(); // Won't clone inner data
    /// *Rc::make_mut(&mut data) += 1;             // Clones inner data
    /// *Rc::make_mut(&mut data) += 1;             // Won't clone anything
    /// *Rc::make_mut(&mut other_data) *= 2;       // Won't clone anything
    ///
    /// // Note: data and other_data now point to different numbers
    /// assert_eq!(*data, 8);
    /// assert_eq!(*other_data, 12);
    ///
    /// ```
    #[inline]
    #[stable(feature = "rc_unique", since = "1.4.0")]
    pub fn make_mut(this: &mut Self) -> &mut T {
        if Rc::strong_count(this) != 1 {
            // Gotta clone the data, there are other Rcs
            *this = Rc::new((**this).clone())
        } else if Rc::weak_count(this) != 0 {
            // Can just steal the data, all that's left is Weaks
            unsafe {
                let mut swap = Rc::new(ptr::read(&(**this._ptr).value));
                mem::swap(this, &mut swap);
                swap.dec_strong();
                // Remove implicit strong-weak ref (no need to craft a fake
                // Weak here -- we know other Weaks can clean up for us)
                swap.dec_weak();
                forget(swap);
            }
        }
        // This unsafety is ok because we're guaranteed that the pointer
        // returned is the *only* pointer that will ever be returned to T. Our
        // reference count is guaranteed to be 1 at this point, and we required
        // the `Rc<T>` itself to be `mut`, so we're returning the only possible
        // reference to the inner value.
        let inner = unsafe { &mut **this._ptr };
        &mut inner.value
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<T: ?Sized> Deref for Rc<T> {
    type Target = T;

    #[inline(always)]
    fn deref(&self) -> &T {
        &self.inner().value
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<T: ?Sized> Drop for Rc<T> {
    /// Drops the `Rc<T>`.
    ///
    /// This will decrement the strong reference count. If the strong reference
    /// count becomes zero and the only other references are `Weak<T>` ones,
    /// `drop`s the inner value.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::rc::Rc;
    ///
    /// {
    ///     let five = Rc::new(5);
    ///
    ///     // stuff
    ///
    ///     drop(five); // explicit drop
    /// }
    /// {
    ///     let five = Rc::new(5);
    ///
    ///     // stuff
    ///
    /// } // implicit drop
    /// ```
    #[unsafe_destructor_blind_to_params]
    fn drop(&mut self) {
        unsafe {
            let ptr = *self._ptr;
            if !(*(&ptr as *const _ as *const *const ())).is_null() &&
               ptr as *const () as usize != mem::POST_DROP_USIZE {
                self.dec_strong();
                if self.strong() == 0 {
                    // destroy the contained object
                    ptr::drop_in_place(&mut (*ptr).value);

                    // remove the implicit "strong weak" pointer now that we've
                    // destroyed the contents.
                    self.dec_weak();

                    if self.weak() == 0 {
                        deallocate(ptr as *mut u8, size_of_val(&*ptr), align_of_val(&*ptr))
                    }
                }
            }
        }
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<T: ?Sized> Clone for Rc<T> {

    /// Makes a clone of the `Rc<T>`.
    ///
    /// When you clone an `Rc<T>`, it will create another pointer to the data and
    /// increase the strong reference counter.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::rc::Rc;
    ///
    /// let five = Rc::new(5);
    ///
    /// five.clone();
    /// ```
    #[inline]
    fn clone(&self) -> Rc<T> {
        self.inc_strong();
        Rc { _ptr: self._ptr }
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<T: Default> Default for Rc<T> {
    /// Creates a new `Rc<T>`, with the `Default` value for `T`.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::rc::Rc;
    ///
    /// let x: Rc<i32> = Default::default();
    /// ```
    #[inline]
    #[stable(feature = "rust1", since = "1.0.0")]
    fn default() -> Rc<T> {
        Rc::new(Default::default())
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<T: ?Sized + PartialEq> PartialEq for Rc<T> {
    /// Equality for two `Rc<T>`s.
    ///
    /// Two `Rc<T>`s are equal if their inner value are equal.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::rc::Rc;
    ///
    /// let five = Rc::new(5);
    ///
    /// five == Rc::new(5);
    /// ```
    #[inline(always)]
    fn eq(&self, other: &Rc<T>) -> bool {
        **self == **other
    }

    /// Inequality for two `Rc<T>`s.
    ///
    /// Two `Rc<T>`s are unequal if their inner value are unequal.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::rc::Rc;
    ///
    /// let five = Rc::new(5);
    ///
    /// five != Rc::new(5);
    /// ```
    #[inline(always)]
    fn ne(&self, other: &Rc<T>) -> bool {
        **self != **other
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<T: ?Sized + Eq> Eq for Rc<T> {}

#[stable(feature = "rust1", since = "1.0.0")]
impl<T: ?Sized + PartialOrd> PartialOrd for Rc<T> {
    /// Partial comparison for two `Rc<T>`s.
    ///
    /// The two are compared by calling `partial_cmp()` on their inner values.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::rc::Rc;
    ///
    /// let five = Rc::new(5);
    ///
    /// five.partial_cmp(&Rc::new(5));
    /// ```
    #[inline(always)]
    fn partial_cmp(&self, other: &Rc<T>) -> Option<Ordering> {
        (**self).partial_cmp(&**other)
    }

    /// Less-than comparison for two `Rc<T>`s.
    ///
    /// The two are compared by calling `<` on their inner values.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::rc::Rc;
    ///
    /// let five = Rc::new(5);
    ///
    /// five < Rc::new(5);
    /// ```
    #[inline(always)]
    fn lt(&self, other: &Rc<T>) -> bool {
        **self < **other
    }

    /// 'Less-than or equal to' comparison for two `Rc<T>`s.
    ///
    /// The two are compared by calling `<=` on their inner values.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::rc::Rc;
    ///
    /// let five = Rc::new(5);
    ///
    /// five <= Rc::new(5);
    /// ```
    #[inline(always)]
    fn le(&self, other: &Rc<T>) -> bool {
        **self <= **other
    }

    /// Greater-than comparison for two `Rc<T>`s.
    ///
    /// The two are compared by calling `>` on their inner values.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::rc::Rc;
    ///
    /// let five = Rc::new(5);
    ///
    /// five > Rc::new(5);
    /// ```
    #[inline(always)]
    fn gt(&self, other: &Rc<T>) -> bool {
        **self > **other
    }

    /// 'Greater-than or equal to' comparison for two `Rc<T>`s.
    ///
    /// The two are compared by calling `>=` on their inner values.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::rc::Rc;
    ///
    /// let five = Rc::new(5);
    ///
    /// five >= Rc::new(5);
    /// ```
    #[inline(always)]
    fn ge(&self, other: &Rc<T>) -> bool {
        **self >= **other
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<T: ?Sized + Ord> Ord for Rc<T> {
    /// Comparison for two `Rc<T>`s.
    ///
    /// The two are compared by calling `cmp()` on their inner values.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::rc::Rc;
    ///
    /// let five = Rc::new(5);
    ///
    /// five.partial_cmp(&Rc::new(5));
    /// ```
    #[inline]
    fn cmp(&self, other: &Rc<T>) -> Ordering {
        (**self).cmp(&**other)
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<T: ?Sized+Hash> Hash for Rc<T> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        (**self).hash(state);
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<T: ?Sized+fmt::Display> fmt::Display for Rc<T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        fmt::Display::fmt(&**self, f)
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<T: ?Sized+fmt::Debug> fmt::Debug for Rc<T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        fmt::Debug::fmt(&**self, f)
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<T> fmt::Pointer for Rc<T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        fmt::Pointer::fmt(&*self._ptr, f)
    }
}

#[stable(feature = "rust1", since = "1.6.0")]
impl<T> From<T> for Rc<T> {
    fn from(t: T) -> Self {
        Rc::new(t)
    }
}

#[stable(feature = "rust1", since = "1.6.0")]
impl<T> From<Box<T>> for Rc<T> {
    fn from(t: Box<T>) -> Self {
        Rc::new(*t)
    }
}

/// A weak version of `Rc<T>`.
///
/// Weak references do not count when determining if the inner value should be
/// dropped.
///
/// See the [module level documentation](./index.html) for more.
#[unsafe_no_drop_flag]
#[stable(feature = "rc_weak", since = "1.4.0")]
pub struct Weak<T: ?Sized> {
    // FIXME #12808: strange names to try to avoid interfering with
    // field accesses of the contained type via Deref
    _ptr: Shared<RcBox<T>>,
}

impl<T: ?Sized> !marker::Send for Weak<T> {}
impl<T: ?Sized> !marker::Sync for Weak<T> {}

#[cfg(not(stage0))] // remove cfg after new snapshot
impl<T: ?Sized+Unsize<U>, U: ?Sized> CoerceUnsized<Weak<U>> for Weak<T> {}

impl<T: ?Sized> Weak<T> {
    /// Upgrades a weak reference to a strong reference.
    ///
    /// Upgrades the `Weak<T>` reference to an `Rc<T>`, if possible.
    ///
    /// Returns `None` if there were no strong references and the data was
    /// destroyed.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::rc::Rc;
    ///
    /// let five = Rc::new(5);
    ///
    /// let weak_five = Rc::downgrade(&five);
    ///
    /// let strong_five: Option<Rc<_>> = weak_five.upgrade();
    /// ```
    #[stable(feature = "rc_weak", since = "1.4.0")]
    pub fn upgrade(&self) -> Option<Rc<T>> {
        if self.strong() == 0 {
            None
        } else {
            self.inc_strong();
            Some(Rc { _ptr: self._ptr })
        }
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<T: ?Sized> Drop for Weak<T> {
    /// Drops the `Weak<T>`.
    ///
    /// This will decrement the weak reference count.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::rc::Rc;
    ///
    /// {
    ///     let five = Rc::new(5);
    ///     let weak_five = Rc::downgrade(&five);
    ///
    ///     // stuff
    ///
    ///     drop(weak_five); // explicit drop
    /// }
    /// {
    ///     let five = Rc::new(5);
    ///     let weak_five = Rc::downgrade(&five);
    ///
    ///     // stuff
    ///
    /// } // implicit drop
    /// ```
    fn drop(&mut self) {
        unsafe {
            let ptr = *self._ptr;
            if !(*(&ptr as *const _ as *const *const ())).is_null() &&
               ptr as *const () as usize != mem::POST_DROP_USIZE {
                self.dec_weak();
                // the weak count starts at 1, and will only go to zero if all
                // the strong pointers have disappeared.
                if self.weak() == 0 {
                    deallocate(ptr as *mut u8, size_of_val(&*ptr), align_of_val(&*ptr))
                }
            }
        }
    }
}

#[stable(feature = "rc_weak", since = "1.4.0")]
impl<T: ?Sized> Clone for Weak<T> {

    /// Makes a clone of the `Weak<T>`.
    ///
    /// This increases the weak reference count.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::rc::Rc;
    ///
    /// let weak_five = Rc::downgrade(&Rc::new(5));
    ///
    /// weak_five.clone();
    /// ```
    #[inline]
    fn clone(&self) -> Weak<T> {
        self.inc_weak();
        Weak { _ptr: self._ptr }
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<T: ?Sized+fmt::Debug> fmt::Debug for Weak<T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "(Weak)")
    }
}

// NOTE: We checked_add here to deal with mem::forget safety. In particular
// if you mem::forget Rcs (or Weaks), the ref-count can overflow, and then
// you can free the allocation while outstanding Rcs (or Weaks) exist.
// We abort because this is such a degenerate scenario that we don't care about
// what happens -- no real program should ever experience this.
//
// This should have negligible overhead since you don't actually need to
// clone these much in Rust thanks to ownership and move-semantics.

#[doc(hidden)]
trait RcBoxPtr<T: ?Sized> {
    fn inner(&self) -> &RcBox<T>;

    #[inline]
    fn strong(&self) -> usize {
        self.inner().strong.get()
    }

    #[inline]
    fn inc_strong(&self) {
        self.inner().strong.set(self.strong().checked_add(1).unwrap_or_else(|| unsafe { abort() }));
    }

    #[inline]
    fn dec_strong(&self) {
        self.inner().strong.set(self.strong() - 1);
    }

    #[inline]
    fn weak(&self) -> usize {
        self.inner().weak.get()
    }

    #[inline]
    fn inc_weak(&self) {
        self.inner().weak.set(self.weak().checked_add(1).unwrap_or_else(|| unsafe { abort() }));
    }

    #[inline]
    fn dec_weak(&self) {
        self.inner().weak.set(self.weak() - 1);
    }
}

impl<T: ?Sized> RcBoxPtr<T> for Rc<T> {
    #[inline(always)]
    fn inner(&self) -> &RcBox<T> {
        unsafe {
            // Safe to assume this here, as if it weren't true, we'd be breaking
            // the contract anyway.
            // This allows the null check to be elided in the destructor if we
            // manipulated the reference count in the same function.
            assume(!(*(&self._ptr as *const _ as *const *const ())).is_null());
            &(**self._ptr)
        }
    }
}

impl<T: ?Sized> RcBoxPtr<T> for Weak<T> {
    #[inline(always)]
    fn inner(&self) -> &RcBox<T> {
        unsafe {
            // Safe to assume this here, as if it weren't true, we'd be breaking
            // the contract anyway.
            // This allows the null check to be elided in the destructor if we
            // manipulated the reference count in the same function.
            assume(!(*(&self._ptr as *const _ as *const *const ())).is_null());
            &(**self._ptr)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{Rc, Weak};
    use std::boxed::Box;
    use std::cell::RefCell;
    use std::option::Option;
    use std::option::Option::{Some, None};
    use std::result::Result::{Err, Ok};
    use std::mem::drop;
    use std::clone::Clone;
    use std::convert::From;

    #[test]
    fn test_clone() {
        let x = Rc::new(RefCell::new(5));
        let y = x.clone();
        *x.borrow_mut() = 20;
        assert_eq!(*y.borrow(), 20);
    }

    #[test]
    fn test_simple() {
        let x = Rc::new(5);
        assert_eq!(*x, 5);
    }

    #[test]
    fn test_simple_clone() {
        let x = Rc::new(5);
        let y = x.clone();
        assert_eq!(*x, 5);
        assert_eq!(*y, 5);
    }

    #[test]
    fn test_destructor() {
        let x: Rc<Box<_>> = Rc::new(box 5);
        assert_eq!(**x, 5);
    }

    #[test]
    fn test_live() {
        let x = Rc::new(5);
        let y = Rc::downgrade(&x);
        assert!(y.upgrade().is_some());
    }

    #[test]
    fn test_dead() {
        let x = Rc::new(5);
        let y = Rc::downgrade(&x);
        drop(x);
        assert!(y.upgrade().is_none());
    }

    #[test]
    fn weak_self_cyclic() {
        struct Cycle {
            x: RefCell<Option<Weak<Cycle>>>,
        }

        let a = Rc::new(Cycle { x: RefCell::new(None) });
        let b = Rc::downgrade(&a.clone());
        *a.x.borrow_mut() = Some(b);

        // hopefully we don't double-free (or leak)...
    }

    #[test]
    fn is_unique() {
        let x = Rc::new(3);
        assert!(Rc::is_unique(&x));
        let y = x.clone();
        assert!(!Rc::is_unique(&x));
        drop(y);
        assert!(Rc::is_unique(&x));
        let w = Rc::downgrade(&x);
        assert!(!Rc::is_unique(&x));
        drop(w);
        assert!(Rc::is_unique(&x));
    }

    #[test]
    fn test_strong_count() {
        let a = Rc::new(0u32);
        assert!(Rc::strong_count(&a) == 1);
        let w = Rc::downgrade(&a);
        assert!(Rc::strong_count(&a) == 1);
        let b = w.upgrade().expect("upgrade of live rc failed");
        assert!(Rc::strong_count(&b) == 2);
        assert!(Rc::strong_count(&a) == 2);
        drop(w);
        drop(a);
        assert!(Rc::strong_count(&b) == 1);
        let c = b.clone();
        assert!(Rc::strong_count(&b) == 2);
        assert!(Rc::strong_count(&c) == 2);
    }

    #[test]
    fn test_weak_count() {
        let a = Rc::new(0u32);
        assert!(Rc::strong_count(&a) == 1);
        assert!(Rc::weak_count(&a) == 0);
        let w = Rc::downgrade(&a);
        assert!(Rc::strong_count(&a) == 1);
        assert!(Rc::weak_count(&a) == 1);
        drop(w);
        assert!(Rc::strong_count(&a) == 1);
        assert!(Rc::weak_count(&a) == 0);
        let c = a.clone();
        assert!(Rc::strong_count(&a) == 2);
        assert!(Rc::weak_count(&a) == 0);
        drop(c);
    }

    #[test]
    fn try_unwrap() {
        let x = Rc::new(3);
        assert_eq!(Rc::try_unwrap(x), Ok(3));
        let x = Rc::new(4);
        let _y = x.clone();
        assert_eq!(Rc::try_unwrap(x), Err(Rc::new(4)));
        let x = Rc::new(5);
        let _w = Rc::downgrade(&x);
        assert_eq!(Rc::try_unwrap(x), Ok(5));
    }

    #[test]
    fn get_mut() {
        let mut x = Rc::new(3);
        *Rc::get_mut(&mut x).unwrap() = 4;
        assert_eq!(*x, 4);
        let y = x.clone();
        assert!(Rc::get_mut(&mut x).is_none());
        drop(y);
        assert!(Rc::get_mut(&mut x).is_some());
        let _w = Rc::downgrade(&x);
        assert!(Rc::get_mut(&mut x).is_none());
    }

    #[test]
    fn test_cowrc_clone_make_unique() {
        let mut cow0 = Rc::new(75);
        let mut cow1 = cow0.clone();
        let mut cow2 = cow1.clone();

        assert!(75 == *Rc::make_mut(&mut cow0));
        assert!(75 == *Rc::make_mut(&mut cow1));
        assert!(75 == *Rc::make_mut(&mut cow2));

        *Rc::make_mut(&mut cow0) += 1;
        *Rc::make_mut(&mut cow1) += 2;
        *Rc::make_mut(&mut cow2) += 3;

        assert!(76 == *cow0);
        assert!(77 == *cow1);
        assert!(78 == *cow2);

        // none should point to the same backing memory
        assert!(*cow0 != *cow1);
        assert!(*cow0 != *cow2);
        assert!(*cow1 != *cow2);
    }

    #[test]
    fn test_cowrc_clone_unique2() {
        let mut cow0 = Rc::new(75);
        let cow1 = cow0.clone();
        let cow2 = cow1.clone();

        assert!(75 == *cow0);
        assert!(75 == *cow1);
        assert!(75 == *cow2);

        *Rc::make_mut(&mut cow0) += 1;

        assert!(76 == *cow0);
        assert!(75 == *cow1);
        assert!(75 == *cow2);

        // cow1 and cow2 should share the same contents
        // cow0 should have a unique reference
        assert!(*cow0 != *cow1);
        assert!(*cow0 != *cow2);
        assert!(*cow1 == *cow2);
    }

    #[test]
    fn test_cowrc_clone_weak() {
        let mut cow0 = Rc::new(75);
        let cow1_weak = Rc::downgrade(&cow0);

        assert!(75 == *cow0);
        assert!(75 == *cow1_weak.upgrade().unwrap());

        *Rc::make_mut(&mut cow0) += 1;

        assert!(76 == *cow0);
        assert!(cow1_weak.upgrade().is_none());
    }

    #[test]
    fn test_show() {
        let foo = Rc::new(75);
        assert_eq!(format!("{:?}", foo), "75");
    }

    #[test]
    fn test_unsized() {
        let foo: Rc<[i32]> = Rc::new([1, 2, 3]);
        assert_eq!(foo, foo.clone());
    }

    #[test]
    fn test_from_owned() {
        let foo = 123;
        let foo_rc = Rc::from(foo);
        assert!(123 == *foo_rc);
    }

    #[test]
    fn test_from_box() {
        let foo_box = Box::new(123);
        let foo_rc = Rc::from(foo_box);
        assert!(123 == *foo_rc);
    }
}

impl<T: ?Sized> borrow::Borrow<T> for Rc<T> {
    fn borrow(&self) -> &T {
        &**self
    }
}

#[stable(since = "1.5.0", feature = "smart_ptr_as_ref")]
impl<T: ?Sized> AsRef<T> for Rc<T> {
    fn as_ref(&self) -> &T {
        &**self
    }
}
