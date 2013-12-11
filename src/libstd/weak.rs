// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

/*! Task-local reference counted boxes with weak pointer support

The `Strong` type is an extension of `std::rc::Rc` with a `downgrade` method returning a `Weak`
pointer type. Ownership of the contained value is shared amongst the `Strong` pointers, and the
value will be destroyed as soon as the last one is gone. A `Weak` pointer can be upgraded to a
`Strong` pointer, but will return `None` if the value has already been freed. It can be used to
avoid creating reference cycles.

For example, a tree with parent pointers can be represented by putting the nodes behind `Strong`
pointers, and then storing the parent pointers as `Weak` pointers.

*/

use cast::transmute;
use ops::Drop;
use cmp::{Eq, Ord};
use clone::{Clone, DeepClone};
use rt::global_heap::exchange_free;
use ptr::read_ptr;
use option::{Option, Some, None};

struct RcBox<T> {
    value: T,
    strong: uint,
    weak: uint
}

/// Immutable reference counted pointer type
#[unsafe_no_drop_flag]
#[no_send]
pub struct Strong<T> {
    priv ptr: *mut RcBox<T>
}

impl<T> Strong<T> {
    /// Construct a new reference-counted box
    pub fn new(value: T) -> Strong<T> {
        unsafe {
            Strong { ptr: transmute(~RcBox { value: value, strong: 1, weak: 0 }) }
        }
    }
}

impl<T> Strong<T> {
    /// Borrow the value contained in the reference-counted box
    #[inline(always)]
    pub fn borrow<'a>(&'a self) -> &'a T {
        unsafe { &(*self.ptr).value }
    }

    /// Downgrade the reference-counted pointer to a weak reference
    pub fn downgrade(&self) -> Weak<T> {
        unsafe {
            (*self.ptr).weak += 1;
            Weak { ptr: self.ptr }
        }
    }
}

#[unsafe_destructor]
impl<T> Drop for Strong<T> {
    fn drop(&mut self) {
        unsafe {
            if self.ptr != 0 as *mut RcBox<T> {
                (*self.ptr).strong -= 1;
                if (*self.ptr).strong == 0 {
                    read_ptr(self.borrow()); // destroy the contained object
                    if (*self.ptr).weak == 0 {
                        exchange_free(self.ptr as *mut u8 as *i8)
                    }
                }
            }
        }
    }
}

impl<T> Clone for Strong<T> {
    #[inline]
    fn clone(&self) -> Strong<T> {
        unsafe {
            (*self.ptr).strong += 1;
            Strong { ptr: self.ptr }
        }
    }
}

impl<T: DeepClone> DeepClone for Strong<T> {
    #[inline]
    fn deep_clone(&self) -> Strong<T> {
        Strong::new(self.borrow().deep_clone())
    }
}

impl<T: Eq> Eq for Strong<T> {
    #[inline(always)]
    fn eq(&self, other: &Strong<T>) -> bool { *self.borrow() == *other.borrow() }

    #[inline(always)]
    fn ne(&self, other: &Strong<T>) -> bool { *self.borrow() != *other.borrow() }
}

impl<T: Ord> Ord for Strong<T> {
    #[inline(always)]
    fn lt(&self, other: &Strong<T>) -> bool { *self.borrow() < *other.borrow() }

    #[inline(always)]
    fn le(&self, other: &Strong<T>) -> bool { *self.borrow() <= *other.borrow() }

    #[inline(always)]
    fn gt(&self, other: &Strong<T>) -> bool { *self.borrow() > *other.borrow() }

    #[inline(always)]
    fn ge(&self, other: &Strong<T>) -> bool { *self.borrow() >= *other.borrow() }
}

/// Weak reference to a reference-counted box
#[unsafe_no_drop_flag]
#[no_send]
pub struct Weak<T> {
    priv ptr: *mut RcBox<T>
}

impl<T> Weak<T> {
    /// Upgrade a weak reference to a strong reference
    pub fn upgrade(&self) -> Option<Strong<T>> {
        unsafe {
            if (*self.ptr).strong == 0 {
                None
            } else {
                (*self.ptr).strong += 1;
                Some(Strong { ptr: self.ptr })
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
                if (*self.ptr).weak == 0 && (*self.ptr).strong == 0 {
                    exchange_free(self.ptr as *mut u8 as *i8)
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
            Weak { ptr: self.ptr }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use prelude::drop;

    #[test]
    fn test_live() {
        let x = Strong::new(5);
        let y = x.downgrade();
        assert!(y.upgrade().is_some());
    }

    #[test]
    fn test_dead() {
        let x = Strong::new(5);
        let y = x.downgrade();
        drop(x);
        assert!(y.upgrade().is_none());
    }
}
