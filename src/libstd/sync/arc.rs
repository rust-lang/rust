// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
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
use sync::atomics::{AtomicUint, SeqCst, Relaxed, Acquire};
use vec;

/// An atomically reference counted pointer.
///
/// Enforces no shared-memory safety.
#[unsafe_no_drop_flag]
pub struct UnsafeArc<T> {
    priv data: *mut ArcData<T>,
}

struct ArcData<T> {
    count: AtomicUint,
    data: T,
}

unsafe fn new_inner<T: Send>(data: T, refcount: uint) -> *mut ArcData<T> {
    let data = ~ArcData { count: AtomicUint::new(refcount), data: data };
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
                vec::from_fn(num_handles, |_| UnsafeArc { data: ptr })
            }
        }
    }

    /// Gets a pointer to the inner shared data. Note that care must be taken to
    /// ensure that the outer `UnsafeArc` does not fall out of scope while this
    /// pointer is in use, otherwise it could possibly contain a use-after-free.
    #[inline]
    pub fn get(&self) -> *mut T {
        unsafe {
            assert!((*self.data).count.load(Relaxed) > 0);
            return &mut (*self.data).data as *mut T;
        }
    }

    /// Gets an immutable pointer to the inner shared data. This has the same
    /// caveats as the `get` method.
    #[inline]
    pub fn get_immut(&self) -> *T {
        unsafe {
            assert!((*self.data).count.load(Relaxed) > 0);
            return &(*self.data).data as *T;
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
            // This barrier might be unnecessary, but I'm not sure...
            let old_count = (*self.data).count.fetch_add(1, Acquire);
            assert!(old_count >= 1);
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
            // Must be acquire+release, not just release, to make sure this
            // doesn't get reordered to after the unwrapper pointer load.
            let old_count = (*self.data).count.fetch_sub(1, SeqCst);
            assert!(old_count >= 1);
            if old_count == 1 {
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
