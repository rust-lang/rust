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

use kinds::Send;
use clone::{Clone, DeepClone};

/// Immutable garbage-collected pointer type
#[no_send]
#[deriving(Clone)]
pub struct Gc<T> {
    priv ptr: @T
}

impl<T: 'static> Gc<T> {
    /// Construct a new garbage-collected box
    #[inline]
    pub fn new(value: T) -> Gc<T> {
        Gc { ptr: @value }
    }
}

impl<T: 'static> Gc<T> {
    /// Borrow the value contained in the garbage-collected box
    #[inline]
    pub fn borrow<'r>(&'r self) -> &'r T {
        &*self.ptr
    }
}

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
    fn test_destructor() {
        let x = Gc::new(~5);
        assert_eq!(**x.borrow(), 5);
    }
}
