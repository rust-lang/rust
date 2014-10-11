// Copyright 2012-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Atomic types
//!
//! Atomic types provide primitive shared-memory communication between
//! threads, and are the building blocks of other concurrent
//! types.
//!
//! This module defines atomic versions of a select number of primitive
//! types, including `AtomicBool`, `AtomicInt`, `AtomicUint`, and `AtomicOption`.
//! Atomic types present operations that, when used correctly, synchronize
//! updates between threads.
//!
//! Each method takes an `Ordering` which represents the strength of
//! the memory barrier for that operation. These orderings are the
//! same as [C++11 atomic orderings][1].
//!
//! [1]: http://gcc.gnu.org/wiki/Atomic/GCCMM/AtomicSync
//!
//! Atomic variables are safe to share between threads (they implement `Sync`)
//! but they do not themselves provide the mechanism for sharing. The most
//! common way to share an atomic variable is to put it into an `Arc` (an
//! atomically-reference-counted shared pointer).
//!
//! Most atomic types may be stored in static variables, initialized using
//! the provided static initializers like `INIT_ATOMIC_BOOL`. Atomic statics
//! are often used for lazy global initialization.
//!
//!
//! # Examples
//!
//! A simple spinlock:
//!
//! ```
//! use std::sync::Arc;
//! use std::sync::atomic::{AtomicUint, SeqCst};
//! use std::task::deschedule;
//!
//! fn main() {
//!     let spinlock = Arc::new(AtomicUint::new(1));
//!
//!     let spinlock_clone = spinlock.clone();
//!     spawn(proc() {
//!         spinlock_clone.store(0, SeqCst);
//!     });
//!
//!     // Wait for the other task to release the lock
//!     while spinlock.load(SeqCst) != 0 {
//!         // Since tasks may not be preemptive (if they are green threads)
//!         // yield to the scheduler to let the other task run. Low level
//!         // concurrent code needs to take into account Rust's two threading
//!         // models.
//!         deschedule();
//!     }
//! }
//! ```
//!
//! Transferring a heap object with `AtomicOption`:
//!
//! ```
//! use std::sync::Arc;
//! use std::sync::atomic::{AtomicOption, SeqCst};
//!
//! fn main() {
//!     struct BigObject;
//!
//!     let shared_big_object = Arc::new(AtomicOption::empty());
//!
//!     let shared_big_object_clone = shared_big_object.clone();
//!     spawn(proc() {
//!         let unwrapped_big_object = shared_big_object_clone.take(SeqCst);
//!         if unwrapped_big_object.is_some() {
//!             println!("got a big object from another task");
//!         } else {
//!             println!("other task hasn't sent big object yet");
//!         }
//!     });
//!
//!     shared_big_object.swap(box BigObject, SeqCst);
//! }
//! ```
//!
//! Keep a global count of live tasks:
//!
//! ```
//! use std::sync::atomic::{AtomicUint, SeqCst, INIT_ATOMIC_UINT};
//!
//! static GLOBAL_TASK_COUNT: AtomicUint = INIT_ATOMIC_UINT;
//!
//! let old_task_count = GLOBAL_TASK_COUNT.fetch_add(1, SeqCst);
//! println!("live tasks: {}", old_task_count + 1);
//! ```

#![allow(deprecated)]

use alloc::boxed::Box;
use core::mem;
use core::prelude::{Drop, None, Option, Some};

pub use core::atomic::{AtomicBool, AtomicInt, AtomicUint, AtomicPtr};
pub use core::atomic::{Ordering, Relaxed, Release, Acquire, AcqRel, SeqCst};
pub use core::atomic::{INIT_ATOMIC_BOOL, INIT_ATOMIC_INT, INIT_ATOMIC_UINT};
pub use core::atomic::fence;

/// An atomic, nullable unique pointer
///
/// This can be used as the concurrency primitive for operations that transfer
/// owned heap objects across tasks.
#[unsafe_no_drop_flag]
#[deprecated = "no longer used; will eventually be replaced by a higher-level\
                concept like MVar"]
pub struct AtomicOption<T> {
    p: AtomicUint,
}

impl<T> AtomicOption<T> {
    /// Create a new `AtomicOption`
    pub fn new(p: Box<T>) -> AtomicOption<T> {
        unsafe { AtomicOption { p: AtomicUint::new(mem::transmute(p)) } }
    }

    /// Create a new `AtomicOption` that doesn't contain a value
    pub fn empty() -> AtomicOption<T> { AtomicOption { p: AtomicUint::new(0) } }

    /// Store a value, returning the old value
    #[inline]
    pub fn swap(&self, val: Box<T>, order: Ordering) -> Option<Box<T>> {
        let val = unsafe { mem::transmute(val) };

        match self.p.swap(val, order) {
            0 => None,
            n => Some(unsafe { mem::transmute(n) }),
        }
    }

    /// Remove the value, leaving the `AtomicOption` empty.
    #[inline]
    pub fn take(&self, order: Ordering) -> Option<Box<T>> {
        unsafe { self.swap(mem::transmute(0u), order) }
    }

    /// Replace an empty value with a non-empty value.
    ///
    /// Succeeds if the option is `None` and returns `None` if so. If
    /// the option was already `Some`, returns `Some` of the rejected
    /// value.
    #[inline]
    pub fn fill(&self, val: Box<T>, order: Ordering) -> Option<Box<T>> {
        unsafe {
            let val = mem::transmute(val);
            let expected = mem::transmute(0u);
            let oldval = self.p.compare_and_swap(expected, val, order);
            if oldval == expected {
                None
            } else {
                Some(mem::transmute(val))
            }
        }
    }

    /// Returns `true` if the `AtomicOption` is empty.
    ///
    /// Be careful: The caller must have some external method of ensuring the
    /// result does not get invalidated by another task after this returns.
    #[inline]
    pub fn is_empty(&self, order: Ordering) -> bool {
        self.p.load(order) as uint == 0
    }
}

#[unsafe_destructor]
impl<T> Drop for AtomicOption<T> {
    fn drop(&mut self) {
        let _ = self.take(SeqCst);
    }
}

#[cfg(test)]
mod test {
    use std::prelude::*;
    use super::*;

    #[test]
    fn option_empty() {
        let option: AtomicOption<()> = AtomicOption::empty();
        assert!(option.is_empty(SeqCst));
    }

    #[test]
    fn option_swap() {
        let p = AtomicOption::new(box 1i);
        let a = box 2i;

        let b = p.swap(a, SeqCst);

        assert!(b == Some(box 1));
        assert!(p.take(SeqCst) == Some(box 2));
    }

    #[test]
    fn option_take() {
        let p = AtomicOption::new(box 1i);

        assert!(p.take(SeqCst) == Some(box 1));
        assert!(p.take(SeqCst) == None);

        let p2 = box 2i;
        p.swap(p2, SeqCst);

        assert!(p.take(SeqCst) == Some(box 2));
    }

    #[test]
    fn option_fill() {
        let p = AtomicOption::new(box 1i);
        assert!(p.fill(box 2i, SeqCst).is_some()); // should fail; shouldn't leak!
        assert!(p.take(SeqCst) == Some(box 1));

        assert!(p.fill(box 2i, SeqCst).is_none()); // shouldn't fail
        assert!(p.take(SeqCst) == Some(box 2));
    }
}
