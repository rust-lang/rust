// Copyright 2012-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

/*!
 * Concurrency-enabled mechanisms for sharing mutable and/or immutable state
 * between tasks.
 */

use std::cast;
use std::ptr;
use std::rt::global_heap;
use std::sync::atomics;

/// An atomically reference counted wrapper for shared state.
///
/// # Example
///
/// In this example, a large vector of floats is shared between several tasks.
/// With simple pipes, without `Arc`, a copy would have to be made for each
/// task.
///
/// ```rust
/// use sync::Arc;
///
/// fn main() {
///     let numbers = Vec::from_fn(100, |i| i as f32);
///     let shared_numbers = Arc::new(numbers);
///
///     for _ in range(0, 10) {
///         let child_numbers = shared_numbers.clone();
///
///         spawn(proc() {
///             let local_numbers = child_numbers.as_slice();
///
///             // Work with the local numbers
///         });
///     }
/// }
/// ```
#[unsafe_no_drop_flag]
pub struct Arc<T> {
    priv x: *mut ArcInner<T>,
}

/// A weak pointer to an `Arc`.
///
/// Weak pointers will not keep the data inside of the `Arc` alive, and can be
/// used to break cycles between `Arc` pointers.
#[unsafe_no_drop_flag]
pub struct Weak<T> {
    priv x: *mut ArcInner<T>,
}

struct ArcInner<T> {
    strong: atomics::AtomicUint,
    weak: atomics::AtomicUint,
    data: T,
}

impl<T: Share + Send> Arc<T> {
    /// Create an atomically reference counted wrapper.
    #[inline]
    pub fn new(data: T) -> Arc<T> {
        // Start the weak pointer count as 1 which is the weak pointer that's
        // held by all the strong pointers (kinda), see std/rc.rs for more info
        let x = ~ArcInner {
            strong: atomics::AtomicUint::new(1),
            weak: atomics::AtomicUint::new(1),
            data: data,
        };
        Arc { x: unsafe { cast::transmute(x) } }
    }

    #[inline]
    fn inner<'a>(&'a self) -> &'a ArcInner<T> {
        // This unsafety is ok because while this arc is alive we're guaranteed
        // that the inner pointer is valid. Furthermore, we know that the
        // `ArcInner` structure itself is `Share` because the inner data is
        // `Share` as well, so we're ok loaning out an immutable pointer to
        // these contents.
        unsafe { &*self.x }
    }

    /// Downgrades a strong pointer to a weak pointer
    ///
    /// Weak pointers will not keep the data alive. Once all strong references
    /// to the underlying data have been dropped, the data itself will be
    /// destroyed.
    pub fn downgrade(&self) -> Weak<T> {
        // See the clone() impl for why this is relaxed
        self.inner().weak.fetch_add(1, atomics::Relaxed);
        Weak { x: self.x }
    }
}

impl<T: Share + Send> Clone for Arc<T> {
    /// Duplicate an atomically reference counted wrapper.
    ///
    /// The resulting two `Arc` objects will point to the same underlying data
    /// object. However, one of the `Arc` objects can be sent to another task,
    /// allowing them to share the underlying data.
    #[inline]
    fn clone(&self) -> Arc<T> {
        // Using a relaxed ordering is alright here, as knowledge of the
        // original reference prevents other threads from erroneously deleting
        // the object.
        //
        // As explained in the [Boost documentation][1], Increasing the
        // reference counter can always be done with memory_order_relaxed: New
        // references to an object can only be formed from an existing
        // reference, and passing an existing reference from one thread to
        // another must already provide any required synchronization.
        //
        // [1]: (www.boost.org/doc/libs/1_55_0/doc/html/atomic/usage_examples.html)
        self.inner().strong.fetch_add(1, atomics::Relaxed);
        Arc { x: self.x }
    }
}

// FIXME(#13042): this should have T: Send, and use self.inner()
impl<T> Deref<T> for Arc<T> {
    #[inline]
    fn deref<'a>(&'a self) -> &'a T {
        let inner = unsafe { &*self.x };
        &inner.data
    }
}

impl<T: Send + Share + Clone> Arc<T> {
    /// Acquires a mutable pointer to the inner contents by guaranteeing that
    /// the reference count is one (no sharing is possible).
    ///
    /// This is also referred to as a copy-on-write operation because the inner
    /// data is cloned if the reference count is greater than one.
    #[inline]
    #[experimental]
    pub fn make_unique<'a>(&'a mut self) -> &'a mut T {
        if self.inner().strong.load(atomics::SeqCst) != 1 {
            *self = Arc::new(self.deref().clone())
        }
        // This unsafety is ok because we're guaranteed that the pointer
        // returned is the *only* pointer that will ever be returned to T. Our
        // reference count is guaranteed to be 1 at this point, and we required
        // the Arc itself to be `mut`, so we're returning the only possible
        // reference to the inner data.
        unsafe { cast::transmute_mut(self.deref()) }
    }
}

#[unsafe_destructor]
impl<T: Share + Send> Drop for Arc<T> {
    fn drop(&mut self) {
        // This structure has #[unsafe_no_drop_flag], so this drop glue may run
        // more than once (but it is guaranteed to be zeroed after the first if
        // it's run more than once)
        if self.x.is_null() { return }

        // Because `fetch_sub` is already atomic, we do not need to synchronize
        // with other threads unless we are going to delete the object. This
        // same logic applies to the below `fetch_sub` to the `weak` count.
        if self.inner().strong.fetch_sub(1, atomics::Release) != 0 { return }

        // This fence is needed to prevent reordering of use of the data and
        // deletion of the data. Because it is marked `Release`, the
        // decreasing of the reference count sychronizes with this `Acquire`
        // fence. This means that use of the data happens before decreasing
        // the refernce count, which happens before this fence, which
        // happens before the deletion of the data.
        //
        // As explained in the [Boost documentation][1],
        //
        // It is important to enforce any possible access to the object in
        // one thread (through an existing reference) to *happen before*
        // deleting the object in a different thread. This is achieved by a
        // "release" operation after dropping a reference (any access to the
        // object through this reference must obviously happened before),
        // and an "acquire" operation before deleting the object.
        //
        // [1]: (www.boost.org/doc/libs/1_55_0/doc/html/atomic/usage_examples.html)
        atomics::fence(atomics::Acquire);

        // Destroy the data at this time, even though we may not free the box
        // allocation itself (there may still be weak pointers lying around).
        unsafe { drop(ptr::read(&self.inner().data)); }

        if self.inner().weak.fetch_sub(1, atomics::Release) == 0 {
            atomics::fence(atomics::Acquire);
            unsafe { global_heap::exchange_free(self.x as *u8) }
        }
    }
}

impl<T: Share + Send> Weak<T> {
    /// Attempts to upgrade this weak reference to a strong reference.
    ///
    /// This method will fail to upgrade this reference if the strong reference
    /// count has already reached 0, but if there are still other active strong
    /// references this function will return a new strong reference to the data
    pub fn upgrade(&self) -> Option<Arc<T>> {
        // We use a CAS loop to increment the strong count instead of a
        // fetch_add because once the count hits 0 is must never be above 0.
        let inner = self.inner();
        loop {
            let n = inner.strong.load(atomics::SeqCst);
            if n == 0 { return None }
            let old = inner.strong.compare_and_swap(n, n + 1, atomics::SeqCst);
            if old == n { return Some(Arc { x: self.x }) }
        }
    }

    #[inline]
    fn inner<'a>(&'a self) -> &'a ArcInner<T> {
        // See comments above for why this is "safe"
        unsafe { &*self.x }
    }
}

impl<T: Share + Send> Clone for Weak<T> {
    #[inline]
    fn clone(&self) -> Weak<T> {
        // See comments in Arc::clone() for why this is relaxed
        self.inner().weak.fetch_add(1, atomics::Relaxed);
        Weak { x: self.x }
    }
}

#[unsafe_destructor]
impl<T: Share + Send> Drop for Weak<T> {
    fn drop(&mut self) {
        // see comments above for why this check is here
        if self.x.is_null() { return }

        // If we find out that we were the last weak pointer, then its time to
        // deallocate the data entirely. See the discussion in Arc::drop() about
        // the memory orderings
        if self.inner().weak.fetch_sub(1, atomics::Release) == 0 {
            atomics::fence(atomics::Acquire);
            unsafe { global_heap::exchange_free(self.x as *u8) }
        }
    }
}

#[cfg(test)]
#[allow(experimental)]
mod tests {
    use super::{Arc, Weak};
    use Mutex;

    use std::task;

    #[test]
    fn manually_share_arc() {
        let v = vec!(1, 2, 3, 4, 5, 6, 7, 8, 9, 10);
        let arc_v = Arc::new(v);

        let (tx, rx) = channel();

        task::spawn(proc() {
            let arc_v: Arc<Vec<int>> = rx.recv();
            assert_eq!(*arc_v.get(3), 4);
        });

        tx.send(arc_v.clone());

        assert_eq!(*arc_v.get(2), 3);
        assert_eq!(*arc_v.get(4), 5);

        info!("{:?}", arc_v);
    }

    #[test]
    fn test_cowarc_clone_make_unique() {
        let mut cow0 = Arc::new(75u);
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
    fn test_cowarc_clone_unique2() {
        let mut cow0 = Arc::new(75u);
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
    fn test_live() {
        let x = Arc::new(5);
        let y = x.downgrade();
        assert!(y.upgrade().is_some());
    }

    #[test]
    fn test_dead() {
        let x = Arc::new(5);
        let y = x.downgrade();
        drop(x);
        assert!(y.upgrade().is_none());
    }

    #[test]
    fn weak_self_cyclic() {
        struct Cycle {
            x: Mutex<Option<Weak<Cycle>>>
        }

        let a = Arc::new(Cycle { x: Mutex::new(None) });
        let b = a.clone().downgrade();
        *a.deref().x.lock().deref_mut() = Some(b);

        // hopefully we don't double-free (or leak)...
    }
}
