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
use core::cmp::{Eq, Ord, TotalEq, TotalOrd, Ordering};
use core::kinds::marker;
use core::ops::{Deref, Drop};
use core::option::{Option, Some, None};
use core::ptr;
use core::ptr::RawPtr;
use core::mem::{min_align_of, size_of};

use heap::deallocate;

struct RcBox<T> {
    value: T,
    strong: Cell<uint>,
    weak: Cell<uint>
}

/// Immutable reference counted pointer type
#[unsafe_no_drop_flag]
pub struct Rc<T> {
    ptr: *mut RcBox<T>,
    nosend: marker::NoSend,
    noshare: marker::NoShare
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
                ptr: transmute(box RcBox {
                    value: value,
                    strong: Cell::new(1),
                    weak: Cell::new(1)
                }),
                nosend: marker::NoSend,
                noshare: marker::NoShare
            }
        }
    }
}

impl<T> Rc<T> {
    /// Downgrade the reference-counted pointer to a weak reference
    pub fn downgrade(&self) -> Weak<T> {
        self.inc_weak();
        Weak {
            ptr: self.ptr,
            nosend: marker::NoSend,
            noshare: marker::NoShare
        }
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
            if !self.ptr.is_null() {
                self.dec_strong();
                if self.strong() == 0 {
                    ptr::read(self.deref()); // destroy the contained object

                    // remove the implicit "strong weak" pointer now
                    // that we've destroyed the contents.
                    self.dec_weak();

                    if self.weak() == 0 {
                        deallocate(self.ptr as *mut u8, size_of::<RcBox<T>>(),
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
        Rc { ptr: self.ptr, nosend: marker::NoSend, noshare: marker::NoShare }
    }
}

impl<T: Eq> Eq for Rc<T> {
    #[inline(always)]
    fn eq(&self, other: &Rc<T>) -> bool { **self == **other }
    #[inline(always)]
    fn ne(&self, other: &Rc<T>) -> bool { **self != **other }
}

impl<T: TotalEq> TotalEq for Rc<T> {}

impl<T: Ord> Ord for Rc<T> {
    #[inline(always)]
    fn lt(&self, other: &Rc<T>) -> bool { **self < **other }

    #[inline(always)]
    fn le(&self, other: &Rc<T>) -> bool { **self <= **other }

    #[inline(always)]
    fn gt(&self, other: &Rc<T>) -> bool { **self > **other }

    #[inline(always)]
    fn ge(&self, other: &Rc<T>) -> bool { **self >= **other }
}

impl<T: TotalOrd> TotalOrd for Rc<T> {
    #[inline]
    fn cmp(&self, other: &Rc<T>) -> Ordering { (**self).cmp(&**other) }
}

/// Weak reference to a reference-counted box
#[unsafe_no_drop_flag]
pub struct Weak<T> {
    ptr: *mut RcBox<T>,
    nosend: marker::NoSend,
    noshare: marker::NoShare
}

impl<T> Weak<T> {
    /// Upgrade a weak reference to a strong reference
    pub fn upgrade(&self) -> Option<Rc<T>> {
        if self.strong() == 0 {
            None
        } else {
            self.inc_strong();
            Some(Rc { ptr: self.ptr, nosend: marker::NoSend, noshare: marker::NoShare })
        }
    }
}

#[unsafe_destructor]
impl<T> Drop for Weak<T> {
    fn drop(&mut self) {
        unsafe {
            if !self.ptr.is_null() {
                self.dec_weak();
                // the weak count starts at 1, and will only go to
                // zero if all the strong pointers have disappeared.
                if self.weak() == 0 {
                    deallocate(self.ptr as *mut u8, size_of::<RcBox<T>>(),
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
        Weak { ptr: self.ptr, nosend: marker::NoSend, noshare: marker::NoShare }
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
    fn inner<'a>(&'a self) -> &'a RcBox<T> { unsafe { &(*self.ptr) } }
}

impl<T> RcBoxPtr<T> for Weak<T> {
    #[inline(always)]
    fn inner<'a>(&'a self) -> &'a RcBox<T> { unsafe { &(*self.ptr) } }
}

#[cfg(test)]
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
}
