// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

/*! Task-local garbage-collected boxes

The `Gc` type provides shared ownership of an immutable value. Destruction is not deterministic, and
will occur some time between every `Gc` handle being gone and the end of the task. The garbage
collector is task-local so `Gc<T>` is not sendable.

*/

#[allow(experimental)];

use kinds::marker;
use kinds::Send;
use clone::{Clone, DeepClone};
use managed;

/// Immutable garbage-collected pointer type
#[lang="gc"]
#[cfg(not(test))]
#[experimental = "Gc is currently based on reference-counting and will not collect cycles until \
                  task annihilation. For now, cycles need to be broken manually by using `Rc<T>` \
                  with a non-owning `Weak<T>` pointer. A tracing garbage collector is planned."]
pub struct Gc<T> {
    priv ptr: @T,
    priv marker: marker::NoSend,
}

#[cfg(test)]
#[no_send]
pub struct Gc<T> {
    priv ptr: @T,
    priv marker: marker::NoSend,
}

impl<T: 'static> Gc<T> {
    /// Construct a new garbage-collected box
    #[inline]
    pub fn new(value: T) -> Gc<T> {
        Gc { ptr: @value, marker: marker::NoSend }
    }

    /// Borrow the value contained in the garbage-collected box
    #[inline]
    pub fn borrow<'r>(&'r self) -> &'r T {
        &*self.ptr
    }

    /// Determine if two garbage-collected boxes point to the same object
    #[inline]
    pub fn ptr_eq(&self, other: &Gc<T>) -> bool {
        managed::ptr_eq(self.ptr, other.ptr)
    }
}

impl<T> Clone for Gc<T> {
    /// Clone the pointer only
    #[inline]
    fn clone(&self) -> Gc<T> {
        Gc{ ptr: self.ptr, marker: marker::NoSend }
    }
}

/// An value that represents the task-local managed heap.
///
/// Use this like `let foo = box(GC) Bar::new(...);`
#[lang="managed_heap"]
#[cfg(not(test))]
pub static GC: () = ();

#[cfg(test)]
pub static GC: () = ();

/// The `Send` bound restricts this to acyclic graphs where it is well-defined.
///
/// A `Freeze` bound would also work, but `Send` *or* `Freeze` cannot be expressed.
impl<T: DeepClone + Send + 'static> DeepClone for Gc<T> {
    #[inline]
    fn deep_clone(&self) -> Gc<T> {
        Gc::new(self.borrow().deep_clone())
    }
}

#[cfg(test)]
mod tests {
    use prelude::*;
    use super::*;
    use cell::RefCell;

    #[test]
    fn test_clone() {
        let x = Gc::new(RefCell::new(5));
        let y = x.clone();
        x.borrow().with_mut(|inner| {
            *inner = 20;
        });
        assert_eq!(y.borrow().with(|x| *x), 20);
    }

    #[test]
    fn test_deep_clone() {
        let x = Gc::new(RefCell::new(5));
        let y = x.deep_clone();
        x.borrow().with_mut(|inner| {
            *inner = 20;
        });
        assert_eq!(y.borrow().with(|x| *x), 5);
    }

    #[test]
    fn test_simple() {
        let x = Gc::new(5);
        assert_eq!(*x.borrow(), 5);
    }

    #[test]
    fn test_simple_clone() {
        let x = Gc::new(5);
        let y = x.clone();
        assert_eq!(*x.borrow(), 5);
        assert_eq!(*y.borrow(), 5);
    }

    #[test]
    fn test_ptr_eq() {
        let x = Gc::new(5);
        let y = x.clone();
        let z = Gc::new(7);
        assert!(x.ptr_eq(&x));
        assert!(x.ptr_eq(&y));
        assert!(!x.ptr_eq(&z));
    }

    #[test]
    fn test_destructor() {
        let x = Gc::new(~5);
        assert_eq!(**x.borrow(), 5);
    }
}
