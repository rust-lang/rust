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

#![allow(experimental)]

use clone::Clone;
use cmp::{Ord, PartialOrd, Ordering, Eq, PartialEq};
use default::Default;
use fmt;
use hash;
use kinds::marker;
use ops::Deref;
use raw;

/// Immutable garbage-collected pointer type
#[lang="gc"]
#[experimental = "Gc is currently based on reference-counting and will not collect cycles until \
                  task annihilation. For now, cycles need to be broken manually by using `Rc<T>` \
                  with a non-owning `Weak<T>` pointer. A tracing garbage collector is planned."]
pub struct Gc<T> {
    _ptr: *mut T,
    marker: marker::NoSend,
}

#[unstable]
impl<T> Clone for Gc<T> {
    /// Clone the pointer only
    #[inline]
    fn clone(&self) -> Gc<T> { *self }
}

/// An value that represents the task-local managed heap.
///
/// Use this like `let foo = box(GC) Bar::new(...);`
#[lang="managed_heap"]
#[cfg(not(test))]
pub static GC: () = ();

impl<T: PartialEq + 'static> PartialEq for Gc<T> {
    #[inline]
    fn eq(&self, other: &Gc<T>) -> bool { *(*self) == *(*other) }
    #[inline]
    fn ne(&self, other: &Gc<T>) -> bool { *(*self) != *(*other) }
}
impl<T: PartialOrd + 'static> PartialOrd for Gc<T> {
    #[inline]
    fn lt(&self, other: &Gc<T>) -> bool { *(*self) < *(*other) }
    #[inline]
    fn le(&self, other: &Gc<T>) -> bool { *(*self) <= *(*other) }
    #[inline]
    fn ge(&self, other: &Gc<T>) -> bool { *(*self) >= *(*other) }
    #[inline]
    fn gt(&self, other: &Gc<T>) -> bool { *(*self) > *(*other) }
}
impl<T: Ord + 'static> Ord for Gc<T> {
    #[inline]
    fn cmp(&self, other: &Gc<T>) -> Ordering { (**self).cmp(&**other) }
}
impl<T: Eq + 'static> Eq for Gc<T> {}

impl<T: 'static> Deref<T> for Gc<T> {
    fn deref<'a>(&'a self) -> &'a T { &**self }
}

impl<T: Default + 'static> Default for Gc<T> {
    fn default() -> Gc<T> {
        box(GC) Default::default()
    }
}

impl<T: 'static> raw::Repr<*const raw::Box<T>> for Gc<T> {}

impl<S: hash::Writer, T: hash::Hash<S> + 'static> hash::Hash<S> for Gc<T> {
    fn hash(&self, s: &mut S) {
        (**self).hash(s)
    }
}

impl<T: 'static + fmt::Show> fmt::Show for Gc<T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        (**self).fmt(f)
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
        *x.borrow().borrow_mut() = 20;
        assert_eq!(*y.borrow().borrow(), 20);
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
        let x = Gc::new(box 5);
        assert_eq!(**x.borrow(), 5);
    }
}
