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

use cast::transmute;
use clone::{Clone, DeepClone};
use cmp::{Eq, Ord};
use kinds::marker;
use ops::{Deref, Drop};
use option::{Option, Some, None};
use ptr;
use rt::global_heap::exchange_free;

struct RcBox<T> {
    value: T,
    strong: uint,
    weak: uint
}

/// Immutable reference counted pointer type
#[unsafe_no_drop_flag]
pub struct Rc<T> {
    priv ptr: *mut RcBox<T>,
    priv marker: marker::NoSend
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
                ptr: transmute(~RcBox { value: value, strong: 1, weak: 1 }),
                marker: marker::NoSend,
            }
        }
    }
}

impl<T> Rc<T> {
    /// Borrow the value contained in the reference-counted box
    #[inline(always)]
    pub fn borrow<'a>(&'a self) -> &'a T {
        unsafe { &(*self.ptr).value }
    }

    /// Downgrade the reference-counted pointer to a weak reference
    pub fn downgrade(&self) -> Weak<T> {
        unsafe {
            (*self.ptr).weak += 1;
            Weak { ptr: self.ptr, marker: marker::NoSend }
        }
    }
}

impl<T> Deref<T> for Rc<T> {
    /// Borrow the value contained in the reference-counted box
    #[inline(always)]
    fn deref<'a>(&'a self) -> &'a T {
        unsafe { &(*self.ptr).value }
    }
}

#[unsafe_destructor]
impl<T> Drop for Rc<T> {
    fn drop(&mut self) {
        unsafe {
            if self.ptr != 0 as *mut RcBox<T> {
                (*self.ptr).strong -= 1;
                if (*self.ptr).strong == 0 {
                    ptr::read(self.borrow()); // destroy the contained object

                    // remove the implicit "strong weak" pointer now
                    // that we've destroyed the contents.
                    (*self.ptr).weak -= 1;

                    if (*self.ptr).weak == 0 {
                        exchange_free(self.ptr as *u8)
                    }
                }
            }
        }
    }
}

impl<T> Clone for Rc<T> {
    #[inline]
    fn clone(&self) -> Rc<T> {
        unsafe {
            (*self.ptr).strong += 1;
            Rc { ptr: self.ptr, marker: marker::NoSend }
        }
    }
}

impl<T: DeepClone> DeepClone for Rc<T> {
    #[inline]
    fn deep_clone(&self) -> Rc<T> {
        Rc::new(self.borrow().deep_clone())
    }
}

impl<T: Eq> Eq for Rc<T> {
    #[inline(always)]
    fn eq(&self, other: &Rc<T>) -> bool { *self.borrow() == *other.borrow() }

    #[inline(always)]
    fn ne(&self, other: &Rc<T>) -> bool { *self.borrow() != *other.borrow() }
}

impl<T: Ord> Ord for Rc<T> {
    #[inline(always)]
    fn lt(&self, other: &Rc<T>) -> bool { *self.borrow() < *other.borrow() }

    #[inline(always)]
    fn le(&self, other: &Rc<T>) -> bool { *self.borrow() <= *other.borrow() }

    #[inline(always)]
    fn gt(&self, other: &Rc<T>) -> bool { *self.borrow() > *other.borrow() }

    #[inline(always)]
    fn ge(&self, other: &Rc<T>) -> bool { *self.borrow() >= *other.borrow() }
}

/// Weak reference to a reference-counted box
#[unsafe_no_drop_flag]
pub struct Weak<T> {
    priv ptr: *mut RcBox<T>,
    priv marker: marker::NoSend
}

impl<T> Weak<T> {
    /// Upgrade a weak reference to a strong reference
    pub fn upgrade(&self) -> Option<Rc<T>> {
        unsafe {
            if (*self.ptr).strong == 0 {
                None
            } else {
                (*self.ptr).strong += 1;
                Some(Rc { ptr: self.ptr, marker: marker::NoSend })
            }
        }
    }
}

#[unsafe_destructor]
impl<T> Drop for Weak<T> {
    fn drop(&mut self) {
        unsafe {
            if self.ptr != 0 as *mut RcBox<T> {
                (*self.ptr).weak -= 1;
                // the weak count starts at 1, and will only go to
                // zero if all the strong pointers have disappeared.
                if (*self.ptr).weak == 0 {
                    exchange_free(self.ptr as *u8)
                }
            }
        }
    }
}

impl<T> Clone for Weak<T> {
    #[inline]
    fn clone(&self) -> Weak<T> {
        unsafe {
            (*self.ptr).weak += 1;
            Weak { ptr: self.ptr, marker: marker::NoSend }
        }
    }
}

#[cfg(test)]
mod tests {
    use prelude::*;
    use super::*;
    use cell::RefCell;

    #[test]
    fn test_clone() {
        let x = Rc::new(RefCell::new(5));
        let y = x.clone();
        x.borrow().with_mut(|inner| {
            *inner = 20;
        });
        assert_eq!(y.borrow().with(|v| *v), 20);
    }

    #[test]
    fn test_deep_clone() {
        let x = Rc::new(RefCell::new(5));
        let y = x.deep_clone();
        x.borrow().with_mut(|inner| {
            *inner = 20;
        });
        assert_eq!(y.borrow().with(|v| *v), 5);
    }

    #[test]
    fn test_simple() {
        let x = Rc::new(5);
        assert_eq!(*x.borrow(), 5);
    }

    #[test]
    fn test_simple_clone() {
        let x = Rc::new(5);
        let y = x.clone();
        assert_eq!(*x.borrow(), 5);
        assert_eq!(*y.borrow(), 5);
    }

    #[test]
    fn test_destructor() {
        let x = Rc::new(~5);
        assert_eq!(**x.borrow(), 5);
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
        use gc::Gc;
        let a = Rc::new(RefCell::new(Gc::new(1)));
        assert!(a.borrow().try_borrow_mut().is_some());
    }

    #[test]
    fn weak_self_cyclic() {
        struct Cycle {
            x: RefCell<Option<Weak<Cycle>>>
        }

        let a = Rc::new(Cycle { x: RefCell::new(None) });
        let b = a.clone().downgrade();
        *a.borrow().x.borrow_mut().get() = Some(b);

        // hopefully we don't double-free (or leak)...
    }
}
