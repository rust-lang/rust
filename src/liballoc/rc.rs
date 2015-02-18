// Copyright 2013-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

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
//!             Owner { name: String::from_str("Gadget Man") }
//!     );
//!
//!     // Create Gadgets belonging to gadget_owner. To increment the reference
//!     // count we clone the `Rc<T>` object.
//!     let gadget1 = Gadget { id: 1, owner: gadget_owner.clone() };
//!     let gadget2 = Gadget { id: 2, owner: gadget_owner.clone() };
//!
//!     drop(gadget_owner);
//!
//!     // Despite dropping gadget_owner, we're still able to print out the name of
//!     // the Owner of the Gadgets. This is because we've only dropped the
//!     // reference count object, not the Owner it wraps. As long as there are
//!     // other `Rc<T>` objects pointing at the same Owner, it will remain allocated. Notice
//!     // that the `Rc<T>` wrapper around Gadget.owner gets automatically dereferenced
//!     // for us.
//!     println!("Gadget {} owned by {}", gadget1.id, gadget1.owner.name);
//!     println!("Gadget {} owned by {}", gadget2.id, gadget2.owner.name);
//!
//!     // At the end of the method, gadget1 and gadget2 get destroyed, and with
//!     // them the last counted references to our Owner. Gadget Man now gets
//!     // destroyed as well.
//! }
//! ```
//!
//! If our requirements change, and we also need to be able to traverse from Owner → Gadget, we
//! will run into problems: an `Rc<T>` pointer from Owner → Gadget introduces a cycle between the
//! objects. This means that their reference counts can never reach 0, and the objects will remain
//! allocated: a memory leak. In order to get around this, we can use `Weak<T>` pointers. These
//! pointers don't contribute to the total count.
//!
//! Rust actually makes it somewhat difficult to produce this loop in the first place: in order to
//! end up with two objects that point at each other, one of them needs to be mutable. This is
//! problematic because `Rc<T>` enforces memory safety by only giving out shared references to the
//! object it wraps, and these don't allow direct mutation. We need to wrap the part of the object
//! we wish to mutate in a `RefCell`, which provides *interior mutability*: a method to achieve
//! mutability through a shared reference. `RefCell` enforces Rust's borrowing rules at runtime.
//! Read the `Cell` documentation for more details on interior mutability.
//!
//! ```rust
//! use std::rc::Rc;
//! use std::rc::Weak;
//! use std::cell::RefCell;
//!
//! struct Owner {
//!     name: String,
//!     gadgets: RefCell<Vec<Weak<Gadget>>>
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
//!     // Create a reference counted Owner. Note the fact that we've put the
//!     // Owner's vector of Gadgets inside a RefCell so that we can mutate it
//!     // through a shared reference.
//!     let gadget_owner : Rc<Owner> = Rc::new(
//!             Owner {
//!                 name: "Gadget Man".to_string(),
//!                 gadgets: RefCell::new(Vec::new())
//!             }
//!     );
//!
//!     // Create Gadgets belonging to gadget_owner as before.
//!     let gadget1 = Rc::new(Gadget{id: 1, owner: gadget_owner.clone()});
//!     let gadget2 = Rc::new(Gadget{id: 2, owner: gadget_owner.clone()});
//!
//!     // Add the Gadgets to their Owner. To do this we mutably borrow from
//!     // the RefCell holding the Owner's Gadgets.
//!     gadget_owner.gadgets.borrow_mut().push(gadget1.clone().downgrade());
//!     gadget_owner.gadgets.borrow_mut().push(gadget2.clone().downgrade());
//!
//!     // Iterate over our Gadgets, printing their details out
//!     for gadget_opt in gadget_owner.gadgets.borrow().iter() {
//!
//!         // gadget_opt is a Weak<Gadget>. Since weak pointers can't guarantee
//!         // that their object is still allocated, we need to call upgrade() on them
//!         // to turn them into a strong reference. This returns an Option, which
//!         // contains a reference to our object if it still exists.
//!         let gadget = gadget_opt.upgrade().unwrap();
//!         println!("Gadget {} owned by {}", gadget.id, gadget.owner.name);
//!     }
//!
//!     // At the end of the method, gadget_owner, gadget1 and gadget2 get
//!     // destroyed. There are now no strong (`Rc<T>`) references to the gadgets.
//!     // Once they get destroyed, the Gadgets get destroyed. This zeroes the
//!     // reference count on Gadget Man, so he gets destroyed as well.
//! }
//! ```

#![stable(feature = "rust1", since = "1.0.0")]

use core::cell::Cell;
use core::clone::Clone;
use core::cmp::{PartialEq, PartialOrd, Eq, Ord, Ordering};
use core::default::Default;
use core::fmt;
use core::hash::{Hasher, Hash};
use core::marker;
use core::mem::{transmute, min_align_of, size_of, forget};
use core::nonzero::NonZero;
use core::ops::{Deref, Drop};
use core::option::Option;
use core::option::Option::{Some, None};
use core::ptr::{self, PtrExt};
use core::result::Result;
use core::result::Result::{Ok, Err};
use core::intrinsics::assume;

use heap::deallocate;

struct RcBox<T> {
    value: T,
    strong: Cell<usize>,
    weak: Cell<usize>
}

/// A reference-counted pointer type over an immutable value.
///
/// See the [module level documentation](./index.html) for more details.
#[unsafe_no_drop_flag]
#[stable(feature = "rust1", since = "1.0.0")]
pub struct Rc<T> {
    // FIXME #12808: strange names to try to avoid interfering with field accesses of the contained
    // type via Deref
    _ptr: NonZero<*mut RcBox<T>>,
}

impl<T> !marker::Send for Rc<T> {}

impl<T> !marker::Sync for Rc<T> {}

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
                // there is an implicit weak pointer owned by all the strong pointers, which
                // ensures that the weak destructor never frees the allocation while the strong
                // destructor is running, even if the weak pointer is stored inside the strong one.
                _ptr: NonZero::new(transmute(box RcBox {
                    value: value,
                    strong: Cell::new(1),
                    weak: Cell::new(1)
                })),
            }
        }
    }

    /// Downgrades the `Rc<T>` to a `Weak<T>` reference.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::rc::Rc;
    ///
    /// let five = Rc::new(5);
    ///
    /// let weak_five = five.downgrade();
    /// ```
    #[unstable(feature = "alloc",
               reason = "Weak pointers may not belong in this module")]
    pub fn downgrade(&self) -> Weak<T> {
        self.inc_weak();
        Weak { _ptr: self._ptr }
    }
}

/// Get the number of weak references to this value.
#[inline]
#[unstable(feature = "alloc")]
pub fn weak_count<T>(this: &Rc<T>) -> usize { this.weak() - 1 }

/// Get the number of strong references to this value.
#[inline]
#[unstable(feature = "alloc")]
pub fn strong_count<T>(this: &Rc<T>) -> usize { this.strong() }

/// Returns true if there are no other `Rc` or `Weak<T>` values that share the same inner value.
///
/// # Examples
///
/// ```
/// use std::rc;
/// use std::rc::Rc;
///
/// let five = Rc::new(5);
///
/// rc::is_unique(&five);
/// ```
#[inline]
#[unstable(feature = "alloc")]
pub fn is_unique<T>(rc: &Rc<T>) -> bool {
    weak_count(rc) == 0 && strong_count(rc) == 1
}

/// Unwraps the contained value if the `Rc<T>` is unique.
///
/// If the `Rc<T>` is not unique, an `Err` is returned with the same `Rc<T>`.
///
/// # Example
///
/// ```
/// use std::rc::{self, Rc};
///
/// let x = Rc::new(3);
/// assert_eq!(rc::try_unwrap(x), Ok(3));
///
/// let x = Rc::new(4);
/// let _y = x.clone();
/// assert_eq!(rc::try_unwrap(x), Err(Rc::new(4)));
/// ```
#[inline]
#[unstable(feature = "alloc")]
pub fn try_unwrap<T>(rc: Rc<T>) -> Result<T, Rc<T>> {
    if is_unique(&rc) {
        unsafe {
            let val = ptr::read(&*rc); // copy the contained object
            // destruct the box and skip our Drop
            // we can ignore the refcounts because we know we're unique
            deallocate(*rc._ptr as *mut u8, size_of::<RcBox<T>>(),
                        min_align_of::<RcBox<T>>());
            forget(rc);
            Ok(val)
        }
    } else {
        Err(rc)
    }
}

/// Returns a mutable reference to the contained value if the `Rc<T>` is unique.
///
/// Returns `None` if the `Rc<T>` is not unique.
///
/// # Example
///
/// ```
/// use std::rc::{self, Rc};
///
/// let mut x = Rc::new(3);
/// *rc::get_mut(&mut x).unwrap() = 4;
/// assert_eq!(*x, 4);
///
/// let _y = x.clone();
/// assert!(rc::get_mut(&mut x).is_none());
/// ```
#[inline]
#[unstable(feature = "alloc")]
pub fn get_mut<'a, T>(rc: &'a mut Rc<T>) -> Option<&'a mut T> {
    if is_unique(rc) {
        let inner = unsafe { &mut **rc._ptr };
        Some(&mut inner.value)
    } else {
        None
    }
}

impl<T: Clone> Rc<T> {
    /// Make a mutable reference from the given `Rc<T>`.
    ///
    /// This is also referred to as a copy-on-write operation because the inner data is cloned if
    /// the reference count is greater than one.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::rc::Rc;
    ///
    /// let mut five = Rc::new(5);
    ///
    /// let mut_five = five.make_unique();
    /// ```
    #[inline]
    #[unstable(feature = "alloc")]
    pub fn make_unique(&mut self) -> &mut T {
        if !is_unique(self) {
            *self = Rc::new((**self).clone())
        }
        // This unsafety is ok because we're guaranteed that the pointer returned is the *only*
        // pointer that will ever be returned to T. Our reference count is guaranteed to be 1 at
        // this point, and we required the `Rc<T>` itself to be `mut`, so we're returning the only
        // possible reference to the inner value.
        let inner = unsafe { &mut **self._ptr };
        &mut inner.value
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<T> Deref for Rc<T> {
    type Target = T;

    #[inline(always)]
    fn deref(&self) -> &T {
        &self.inner().value
    }
}

#[unsafe_destructor]
#[stable(feature = "rust1", since = "1.0.0")]
impl<T> Drop for Rc<T> {
    /// Drops the `Rc<T>`.
    ///
    /// This will decrement the strong reference count. If the strong reference count becomes zero
    /// and the only other references are `Weak<T>` ones, `drop`s the inner value.
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
    fn drop(&mut self) {
        unsafe {
            let ptr = *self._ptr;
            if !ptr.is_null() {
                self.dec_strong();
                if self.strong() == 0 {
                    ptr::read(&**self); // destroy the contained object

                    // remove the implicit "strong weak" pointer now that we've destroyed the
                    // contents.
                    self.dec_weak();

                    if self.weak() == 0 {
                        deallocate(ptr as *mut u8, size_of::<RcBox<T>>(),
                                   min_align_of::<RcBox<T>>())
                    }
                }
            }
        }
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<T> Clone for Rc<T> {

    /// Makes a clone of the `Rc<T>`.
    ///
    /// This increases the strong reference count.
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
    /// use std::default::Default;
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
impl<T: PartialEq> PartialEq for Rc<T> {
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
    fn eq(&self, other: &Rc<T>) -> bool { **self == **other }

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
    fn ne(&self, other: &Rc<T>) -> bool { **self != **other }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<T: Eq> Eq for Rc<T> {}

#[stable(feature = "rust1", since = "1.0.0")]
impl<T: PartialOrd> PartialOrd for Rc<T> {
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
    fn lt(&self, other: &Rc<T>) -> bool { **self < **other }

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
    fn le(&self, other: &Rc<T>) -> bool { **self <= **other }

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
    fn gt(&self, other: &Rc<T>) -> bool { **self > **other }

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
    fn ge(&self, other: &Rc<T>) -> bool { **self >= **other }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<T: Ord> Ord for Rc<T> {
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
    fn cmp(&self, other: &Rc<T>) -> Ordering { (**self).cmp(&**other) }
}

// FIXME (#18248) Make `T` `Sized?`
#[cfg(stage0)]
impl<S: Hasher, T: Hash<S>> Hash<S> for Rc<T> {
    #[inline]
    fn hash(&self, state: &mut S) {
        (**self).hash(state);
    }
}
#[cfg(not(stage0))]
#[stable(feature = "rust1", since = "1.0.0")]
impl<T: Hash> Hash for Rc<T> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        (**self).hash(state);
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<T: fmt::Display> fmt::Display for Rc<T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        fmt::Display::fmt(&**self, f)
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<T: fmt::Debug> fmt::Debug for Rc<T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        fmt::Debug::fmt(&**self, f)
    }
}

/// A weak version of `Rc<T>`.
///
/// Weak references do not count when determining if the inner value should be dropped.
///
/// See the [module level documentation](./index.html) for more.
#[unsafe_no_drop_flag]
#[unstable(feature = "alloc",
           reason = "Weak pointers may not belong in this module.")]
pub struct Weak<T> {
    // FIXME #12808: strange names to try to avoid interfering with
    // field accesses of the contained type via Deref
    _ptr: NonZero<*mut RcBox<T>>,
}

impl<T> !marker::Send for Weak<T> {}

impl<T> !marker::Sync for Weak<T> {}


#[unstable(feature = "alloc",
           reason = "Weak pointers may not belong in this module.")]
impl<T> Weak<T> {

    /// Upgrades a weak reference to a strong reference.
    ///
    /// Upgrades the `Weak<T>` reference to an `Rc<T>`, if possible.
    ///
    /// Returns `None` if there were no strong references and the data was destroyed.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::rc::Rc;
    ///
    /// let five = Rc::new(5);
    ///
    /// let weak_five = five.downgrade();
    ///
    /// let strong_five: Option<Rc<_>> = weak_five.upgrade();
    /// ```
    pub fn upgrade(&self) -> Option<Rc<T>> {
        if self.strong() == 0 {
            None
        } else {
            self.inc_strong();
            Some(Rc { _ptr: self._ptr })
        }
    }
}

#[unsafe_destructor]
#[stable(feature = "rust1", since = "1.0.0")]
impl<T> Drop for Weak<T> {
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
    ///     let weak_five = five.downgrade();
    ///
    ///     // stuff
    ///
    ///     drop(weak_five); // explicit drop
    /// }
    /// {
    ///     let five = Rc::new(5);
    ///     let weak_five = five.downgrade();
    ///
    ///     // stuff
    ///
    /// } // implicit drop
    /// ```
    fn drop(&mut self) {
        unsafe {
            let ptr = *self._ptr;
            if !ptr.is_null() {
                self.dec_weak();
                // the weak count starts at 1, and will only go to zero if all the strong pointers
                // have disappeared.
                if self.weak() == 0 {
                    deallocate(ptr as *mut u8, size_of::<RcBox<T>>(),
                               min_align_of::<RcBox<T>>())
                }
            }
        }
    }
}

#[unstable(feature = "alloc",
           reason = "Weak pointers may not belong in this module.")]
impl<T> Clone for Weak<T> {

    /// Makes a clone of the `Weak<T>`.
    ///
    /// This increases the weak reference count.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::rc::Rc;
    ///
    /// let weak_five = Rc::new(5).downgrade();
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
impl<T: fmt::Debug> fmt::Debug for Weak<T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "(Weak)")
    }
}

#[doc(hidden)]
trait RcBoxPtr<T> {
    fn inner(&self) -> &RcBox<T>;

    #[inline]
    fn strong(&self) -> usize { self.inner().strong.get() }

    #[inline]
    fn inc_strong(&self) { self.inner().strong.set(self.strong() + 1); }

    #[inline]
    fn dec_strong(&self) { self.inner().strong.set(self.strong() - 1); }

    #[inline]
    fn weak(&self) -> usize { self.inner().weak.get() }

    #[inline]
    fn inc_weak(&self) { self.inner().weak.set(self.weak() + 1); }

    #[inline]
    fn dec_weak(&self) { self.inner().weak.set(self.weak() - 1); }
}

impl<T> RcBoxPtr<T> for Rc<T> {
    #[inline(always)]
    fn inner(&self) -> &RcBox<T> {
        unsafe {
            // Safe to assume this here, as if it weren't true, we'd be breaking
            // the contract anyway.
            // This allows the null check to be elided in the destructor if we
            // manipulated the reference count in the same function.
            assume(!self._ptr.is_null());
            &(**self._ptr)
        }
    }
}

impl<T> RcBoxPtr<T> for Weak<T> {
    #[inline(always)]
    fn inner(&self) -> &RcBox<T> {
        unsafe {
            // Safe to assume this here, as if it weren't true, we'd be breaking
            // the contract anyway.
            // This allows the null check to be elided in the destructor if we
            // manipulated the reference count in the same function.
            assume(!self._ptr.is_null());
            &(**self._ptr)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{Rc, Weak, weak_count, strong_count};
    use std::cell::RefCell;
    use std::option::Option;
    use std::option::Option::{Some, None};
    use std::result::Result::{Err, Ok};
    use std::mem::drop;
    use std::clone::Clone;

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
        let x = Rc::new(box 5);
        assert_eq!(**x, 5);
    }

    #[test]
    fn test_live() {
        let x = Rc::new(5);
        let y = x.downgrade();
        assert!(y.upgrade().is_some());
    }

    #[test]
    fn test_dead() {
        let x = Rc::new(5);
        let y = x.downgrade();
        drop(x);
        assert!(y.upgrade().is_none());
    }

    #[test]
    fn weak_self_cyclic() {
        struct Cycle {
            x: RefCell<Option<Weak<Cycle>>>
        }

        let a = Rc::new(Cycle { x: RefCell::new(None) });
        let b = a.clone().downgrade();
        *a.x.borrow_mut() = Some(b);

        // hopefully we don't double-free (or leak)...
    }

    #[test]
    fn is_unique() {
        let x = Rc::new(3);
        assert!(super::is_unique(&x));
        let y = x.clone();
        assert!(!super::is_unique(&x));
        drop(y);
        assert!(super::is_unique(&x));
        let w = x.downgrade();
        assert!(!super::is_unique(&x));
        drop(w);
        assert!(super::is_unique(&x));
    }

    #[test]
    fn test_strong_count() {
        let a = Rc::new(0u32);
        assert!(strong_count(&a) == 1);
        let w = a.downgrade();
        assert!(strong_count(&a) == 1);
        let b = w.upgrade().expect("upgrade of live rc failed");
        assert!(strong_count(&b) == 2);
        assert!(strong_count(&a) == 2);
        drop(w);
        drop(a);
        assert!(strong_count(&b) == 1);
        let c = b.clone();
        assert!(strong_count(&b) == 2);
        assert!(strong_count(&c) == 2);
    }

    #[test]
    fn test_weak_count() {
        let a = Rc::new(0u32);
        assert!(strong_count(&a) == 1);
        assert!(weak_count(&a) == 0);
        let w = a.downgrade();
        assert!(strong_count(&a) == 1);
        assert!(weak_count(&a) == 1);
        drop(w);
        assert!(strong_count(&a) == 1);
        assert!(weak_count(&a) == 0);
        let c = a.clone();
        assert!(strong_count(&a) == 2);
        assert!(weak_count(&a) == 0);
        drop(c);
    }

    #[test]
    fn try_unwrap() {
        let x = Rc::new(3);
        assert_eq!(super::try_unwrap(x), Ok(3));
        let x = Rc::new(4);
        let _y = x.clone();
        assert_eq!(super::try_unwrap(x), Err(Rc::new(4)));
        let x = Rc::new(5);
        let _w = x.downgrade();
        assert_eq!(super::try_unwrap(x), Err(Rc::new(5)));
    }

    #[test]
    fn get_mut() {
        let mut x = Rc::new(3);
        *super::get_mut(&mut x).unwrap() = 4;
        assert_eq!(*x, 4);
        let y = x.clone();
        assert!(super::get_mut(&mut x).is_none());
        drop(y);
        assert!(super::get_mut(&mut x).is_some());
        let _w = x.downgrade();
        assert!(super::get_mut(&mut x).is_none());
    }

    #[test]
    fn test_cowrc_clone_make_unique() {
        let mut cow0 = Rc::new(75);
        let mut cow1 = cow0.clone();
        let mut cow2 = cow1.clone();

        assert!(75 == *cow0.make_unique());
        assert!(75 == *cow1.make_unique());
        assert!(75 == *cow2.make_unique());

        *cow0.make_unique() += 1;
        *cow1.make_unique() += 2;
        *cow2.make_unique() += 3;

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

        *cow0.make_unique() += 1;

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
        let cow1_weak = cow0.downgrade();

        assert!(75 == *cow0);
        assert!(75 == *cow1_weak.upgrade().unwrap());

        *cow0.make_unique() += 1;

        assert!(76 == *cow0);
        assert!(cow1_weak.upgrade().is_none());
    }

    #[test]
    fn test_show() {
        let foo = Rc::new(75);
        assert_eq!(format!("{:?}", foo), "75");
    }

}
