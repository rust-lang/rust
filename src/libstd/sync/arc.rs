// Copyright 2013-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Atomically reference counted data
//!
//! This modules contains the implementation of an atomically reference counted
//! pointer for the purpose of sharing data between tasks. This is obviously a
//! very unsafe primitive to use, but it has its use cases when implementing
//! concurrent data structures and similar tasks.
//!
//! Great care must be taken to ensure that data races do not arise through the
//! usage of `UnsafeArc`, and this often requires some form of external
//! synchronization. The only guarantee provided to you by this class is that
//! the underlying data will remain valid (not free'd) so long as the reference
//! count is greater than one.

use cast;
use clone::Clone;
use kinds::Send;
use ops::Drop;
use ptr::RawPtr;
use sync::atomics::{fence, AtomicUint, Relaxed, Acquire, Release};
use slice;
use ty::Unsafe;

/// An atomically reference counted pointer.
///
/// Enforces no shared-memory safety.
#[unsafe_no_drop_flag]
pub struct UnsafeArc<T> {
    data: *mut ArcData<T>,
}

struct ArcData<T> {
    count: AtomicUint,
    data: Unsafe<T>,
}

unsafe fn new_inner<T: Send>(data: T, refcount: uint) -> *mut ArcData<T> {
    let data = ~ArcData {
                    count: AtomicUint::new(refcount),
                    data: Unsafe::new(data)
                 };
    cast::transmute(data)
}

impl<T: Send> UnsafeArc<T> {
    /// Creates a new `UnsafeArc` which wraps the given data.
    pub fn new(data: T) -> UnsafeArc<T> {
        unsafe { UnsafeArc { data: new_inner(data, 1) } }
    }

    /// As new(), but returns an extra pre-cloned handle.
    pub fn new2(data: T) -> (UnsafeArc<T>, UnsafeArc<T>) {
        unsafe {
            let ptr = new_inner(data, 2);
            (UnsafeArc { data: ptr }, UnsafeArc { data: ptr })
        }
    }

    /// As new(), but returns a vector of as many pre-cloned handles as
    /// requested.
    pub fn newN(data: T, num_handles: uint) -> ~[UnsafeArc<T>] {
        unsafe {
            if num_handles == 0 {
                ~[] // need to free data here
            } else {
                let ptr = new_inner(data, num_handles);
                slice::from_fn(num_handles, |_| UnsafeArc { data: ptr })
            }
        }
    }

    /// Gets a pointer to the inner shared data. Note that care must be taken to
    /// ensure that the outer `UnsafeArc` does not fall out of scope while this
    /// pointer is in use, otherwise it could possibly contain a use-after-free.
    #[inline]
    pub fn get(&self) -> *mut T {
        unsafe {
            // FIXME(#12049): this needs some sort of debug assertion
            if cfg!(test) { assert!((*self.data).count.load(Relaxed) > 0); }
            return (*self.data).data.get();
        }
    }

    /// Gets an immutable pointer to the inner shared data. This has the same
    /// caveats as the `get` method.
    #[inline]
    pub fn get_immut(&self) -> *T {
        unsafe {
            // FIXME(#12049): this needs some sort of debug assertion
            if cfg!(test) { assert!((*self.data).count.load(Relaxed) > 0); }
            return (*self.data).data.get() as *T;
        }
    }

    /// checks if this is the only reference to the arc protected data
    #[inline]
    pub fn is_owned(&self) -> bool {
        unsafe {
            (*self.data).count.load(Relaxed) == 1
        }
    }
}

impl<T: Send> Clone for UnsafeArc<T> {
    fn clone(&self) -> UnsafeArc<T> {
        unsafe {
            // Using a relaxed ordering is alright here, as knowledge of the original reference
            // prevents other threads from erroneously deleting the object.
            //
            // As explained in the [Boost documentation][1],
            //  Increasing the reference counter can always be done with memory_order_relaxed: New
            //  references to an object can only be formed from an existing reference, and passing
            //  an existing reference from one thread to another must already provide any required
            //  synchronization.
            // [1]: (www.boost.org/doc/libs/1_55_0/doc/html/atomic/usage_examples.html)
            let old_count = (*self.data).count.fetch_add(1, Relaxed);
            // FIXME(#12049): this needs some sort of debug assertion
            if cfg!(test) { assert!(old_count >= 1); }
            return UnsafeArc { data: self.data };
        }
    }
}

#[unsafe_destructor]
impl<T> Drop for UnsafeArc<T>{
    fn drop(&mut self) {
        unsafe {
            // Happens when destructing an unwrapper's handle and from
            // `#[unsafe_no_drop_flag]`
            if self.data.is_null() {
                return
            }
            // Because `fetch_sub` is already atomic, we do not need to synchronize with other
            // threads unless we are going to delete the object.
            let old_count = (*self.data).count.fetch_sub(1, Release);
            // FIXME(#12049): this needs some sort of debug assertion
            if cfg!(test) { assert!(old_count >= 1); }
            if old_count == 1 {
                // This fence is needed to prevent reordering of use of the data and deletion of
                // the data. Because it is marked `Release`, the decreasing of the reference count
                // sychronizes with this `Acquire` fence. This means that use of the data happens
                // before decreasing the refernce count, which happens before this fence, which
                // happens before the deletion of the data.
                //
                // As explained in the [Boost documentation][1],
                //  It is important to enforce any possible access to the object in one thread
                //  (through an existing reference) to *happen before* deleting the object in a
                //  different thread. This is achieved by a "release" operation after dropping a
                //  reference (any access to the object through this reference must obviously
                //  happened before), and an "acquire" operation before deleting the object.
                // [1]: (www.boost.org/doc/libs/1_55_0/doc/html/atomic/usage_examples.html)
                fence(Acquire);
                let _: ~ArcData<T> = cast::transmute(self.data);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use prelude::*;
    use super::UnsafeArc;
    use mem::size_of;

    #[test]
    fn test_size() {
        assert_eq!(size_of::<UnsafeArc<[int, ..10]>>(), size_of::<*[int, ..10]>());
    }

    #[test]
    fn arclike_newN() {
        // Tests that the many-refcounts-at-once constructors don't leak.
        let _ = UnsafeArc::new2(~~"hello");
        let x = UnsafeArc::newN(~~"hello", 0);
        assert_eq!(x.len(), 0)
        let x = UnsafeArc::newN(~~"hello", 1);
        assert_eq!(x.len(), 1)
        let x = UnsafeArc::newN(~~"hello", 10);
        assert_eq!(x.len(), 10)
    }
}
