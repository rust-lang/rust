// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

/*! Task-local reference-counted boxes (`Rc` type)

The `Rc` type provides shared ownership of an immutable value. Destruction is deterministic, and
will occur as soon as the last owner is gone. It is marked as non-sendable because it avoids the
overhead of atomic reference counting.

The `downgrade` method can be used to create a non-owning `Weak` pointer to the box. A `Weak`
pointer can be upgraded to an `Rc` pointer, but will return `None` if the value has already been
freed.

For example, a tree with parent pointers can be represented by putting the nodes behind `Strong`
pointers, and then storing the parent pointers as `Weak` pointers.

*/

use core::mem::transmute;
use core::cell::Cell;
use core::clone::Clone;
use core::cmp::{PartialEq, PartialOrd, Eq, Ord, Ordering};
use core::kinds::marker;
use core::ops::{Deref, Drop};
use core::option::{Option, Some, None};
use core::ptr;
use core::ptr::RawPtr;
use core::mem::{min_align_of, size_of};
use core::fmt;

use heap::deallocate;

struct RcBox<T> {
    value: T,
    strong: Cell<uint>,
    weak: Cell<uint>
}

/// Immutable reference counted pointer type
#[unsafe_no_drop_flag]
pub struct Rc<T> {
    // FIXME #12808: strange names to try to avoid interfering with
    // field accesses of the contained type via Deref
    _ptr: *mut RcBox<T>,
    _nosend: marker::NoSend,
    _noshare: marker::NoShare
}

impl<T> Rc<T> {
    /// Construct a new reference-counted box
    pub fn new(value: T) -> Rc<T> {
        unsafe {
            Rc {
                // there is an implicit weak pointer owned by all the
                // strong pointers, which ensures that the weak
                // destructor never frees the allocation while the
                // strong destructor is running, even if the weak
                // pointer is stored inside the strong one.
                _ptr: transmute(box RcBox {
                    value: value,
                    strong: Cell::new(1),
                    weak: Cell::new(1)
                }),
                _nosend: marker::NoSend,
                _noshare: marker::NoShare
            }
        }
    }
}

impl<T> Rc<T> {
    /// Downgrade the reference-counted pointer to a weak reference
    pub fn downgrade(&self) -> Weak<T> {
        self.inc_weak();
        Weak {
            _ptr: self._ptr,
            _nosend: marker::NoSend,
            _noshare: marker::NoShare
        }
    }
}

impl<T: Clone> Rc<T> {
    /// Acquires a mutable pointer to the inner contents by guaranteeing that
    /// the reference count is one (no sharing is possible).
    ///
    /// This is also referred to as a copy-on-write operation because the inner
    /// data is cloned if the reference count is greater than one.
    #[inline]
    #[experimental]
    pub fn make_unique<'a>(&'a mut self) -> &'a mut T {
        // Note that we hold a strong reference, which also counts as
        // a weak reference, so we only clone if there is an
        // additional reference of either kind.
        if self.strong() != 1 || self.weak() != 1 {
            *self = Rc::new(self.deref().clone())
        }
        // This unsafety is ok because we're guaranteed that the pointer
        // returned is the *only* pointer that will ever be returned to T. Our
        // reference count is guaranteed to be 1 at this point, and we required
        // the Rc itself to be `mut`, so we're returning the only possible
        // reference to the inner data.
        let inner = unsafe { &mut *self._ptr };
        &mut inner.value
    }
}

impl<T> Deref<T> for Rc<T> {
    /// Borrow the value contained in the reference-counted box
    #[inline(always)]
    fn deref<'a>(&'a self) -> &'a T {
        &self.inner().value
    }
}

#[unsafe_destructor]
impl<T> Drop for Rc<T> {
    fn drop(&mut self) {
        unsafe {
            if !self._ptr.is_null() {
                self.dec_strong();
                if self.strong() == 0 {
                    ptr::read(self.deref()); // destroy the contained object

                    // remove the implicit "strong weak" pointer now
                    // that we've destroyed the contents.
                    self.dec_weak();

                    if self.weak() == 0 {
                        deallocate(self._ptr as *mut u8, size_of::<RcBox<T>>(),
                                   min_align_of::<RcBox<T>>())
                    }
                }
            }
        }
    }
}

impl<T> Clone for Rc<T> {
    #[inline]
    fn clone(&self) -> Rc<T> {
        self.inc_strong();
        Rc { _ptr: self._ptr, _nosend: marker::NoSend, _noshare: marker::NoShare }
    }
}

impl<T: PartialEq> PartialEq for Rc<T> {
    #[inline(always)]
    fn eq(&self, other: &Rc<T>) -> bool { **self == **other }
    #[inline(always)]
    fn ne(&self, other: &Rc<T>) -> bool { **self != **other }
}

impl<T: Eq> Eq for Rc<T> {}

impl<T: PartialOrd> PartialOrd for Rc<T> {
    #[inline(always)]
    fn lt(&self, other: &Rc<T>) -> bool { **self < **other }

    #[inline(always)]
    fn le(&self, other: &Rc<T>) -> bool { **self <= **other }

    #[inline(always)]
    fn gt(&self, other: &Rc<T>) -> bool { **self > **other }

    #[inline(always)]
    fn ge(&self, other: &Rc<T>) -> bool { **self >= **other }
}

impl<T: Ord> Ord for Rc<T> {
    #[inline]
    fn cmp(&self, other: &Rc<T>) -> Ordering { (**self).cmp(&**other) }
}

impl<T: fmt::Show> fmt::Show for Rc<T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        (**self).fmt(f)
    }
}

/// Weak reference to a reference-counted box
#[unsafe_no_drop_flag]
pub struct Weak<T> {
    // FIXME #12808: strange names to try to avoid interfering with
    // field accesses of the contained type via Deref
    _ptr: *mut RcBox<T>,
    _nosend: marker::NoSend,
    _noshare: marker::NoShare
}

impl<T> Weak<T> {
    /// Upgrade a weak reference to a strong reference
    pub fn upgrade(&self) -> Option<Rc<T>> {
        if self.strong() == 0 {
            None
        } else {
            self.inc_strong();
            Some(Rc { _ptr: self._ptr, _nosend: marker::NoSend, _noshare: marker::NoShare })
        }
    }
}

#[unsafe_destructor]
impl<T> Drop for Weak<T> {
    fn drop(&mut self) {
        unsafe {
            if !self._ptr.is_null() {
                self.dec_weak();
                // the weak count starts at 1, and will only go to
                // zero if all the strong pointers have disappeared.
                if self.weak() == 0 {
                    deallocate(self._ptr as *mut u8, size_of::<RcBox<T>>(),
                               min_align_of::<RcBox<T>>())
                }
            }
        }
    }
}

impl<T> Clone for Weak<T> {
    #[inline]
    fn clone(&self) -> Weak<T> {
        self.inc_weak();
        Weak { _ptr: self._ptr, _nosend: marker::NoSend, _noshare: marker::NoShare }
    }
}

#[doc(hidden)]
trait RcBoxPtr<T> {
    fn inner<'a>(&'a self) -> &'a RcBox<T>;

    #[inline]
    fn strong(&self) -> uint { self.inner().strong.get() }

    #[inline]
    fn inc_strong(&self) { self.inner().strong.set(self.strong() + 1); }

    #[inline]
    fn dec_strong(&self) { self.inner().strong.set(self.strong() - 1); }

    #[inline]
    fn weak(&self) -> uint { self.inner().weak.get() }

    #[inline]
    fn inc_weak(&self) { self.inner().weak.set(self.weak() + 1); }

    #[inline]
    fn dec_weak(&self) { self.inner().weak.set(self.weak() - 1); }
}

impl<T> RcBoxPtr<T> for Rc<T> {
    #[inline(always)]
    fn inner<'a>(&'a self) -> &'a RcBox<T> { unsafe { &(*self._ptr) } }
}

impl<T> RcBoxPtr<T> for Weak<T> {
    #[inline(always)]
    fn inner<'a>(&'a self) -> &'a RcBox<T> { unsafe { &(*self._ptr) } }
}

#[cfg(test)]
#[allow(experimental)]
mod tests {
    use super::{Rc, Weak};
    use std::cell::RefCell;
    use std::option::{Option, Some, None};
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
    fn gc_inside() {
        // see issue #11532
        use std::gc::Gc;
        let a = Rc::new(RefCell::new(Gc::new(1)));
        assert!(a.try_borrow_mut().is_some());
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
    fn test_cowrc_clone_make_unique() {
        let mut cow0 = Rc::new(75u);
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
        let mut cow0 = Rc::new(75u);
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
        let mut cow0 = Rc::new(75u);
        let cow1_weak = cow0.downgrade();

        assert!(75 == *cow0);
        assert!(75 == *cow1_weak.upgrade().unwrap());

        *cow0.make_unique() += 1;

        assert!(76 == *cow0);
        assert!(cow1_weak.upgrade().is_none());
    }

}
