// Copyright 2012-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![stable(feature = "rust1", since = "1.0.0")]

//! Threadsafe reference-counted boxes (the `Arc<T>` type).
//!
//! The `Arc<T>` type provides shared ownership of an immutable value.
//! Destruction is deterministic, and will occur as soon as the last owner is
//! gone. It is marked as `Send` because it uses atomic reference counting.
//!
//! If you do not need thread-safety, and just need shared ownership, consider
//! the [`Rc<T>` type](../rc/struct.Rc.html). It is the same as `Arc<T>`, but
//! does not use atomics, making it both thread-unsafe as well as significantly
//! faster when updating the reference count.
//!
//! The `downgrade` method can be used to create a non-owning `Weak<T>` pointer
//! to the box. A `Weak<T>` pointer can be upgraded to an `Arc<T>` pointer, but
//! will return `None` if the value has already been dropped.
//!
//! For example, a tree with parent pointers can be represented by putting the
//! nodes behind strong `Arc<T>` pointers, and then storing the parent pointers
//! as `Weak<T>` pointers.
//!
//! # Examples
//!
//! Sharing some immutable data between threads:
//!
//! ```no_run
//! use std::sync::Arc;
//! use std::thread;
//!
//! let five = Arc::new(5);
//!
//! for _ in 0..10 {
//!     let five = five.clone();
//!
//!     thread::spawn(move || {
//!         println!("{:?}", five);
//!     });
//! }
//! ```
//!
//! Sharing mutable data safely between threads with a `Mutex`:
//!
//! ```no_run
//! use std::sync::{Arc, Mutex};
//! use std::thread;
//!
//! let five = Arc::new(Mutex::new(5));
//!
//! for _ in 0..10 {
//!     let five = five.clone();
//!
//!     thread::spawn(move || {
//!         let mut number = five.lock().unwrap();
//!
//!         *number += 1;
//!
//!         println!("{}", *number); // prints 6
//!     });
//! }
//! ```

use boxed::Box;

use core::sync::atomic;
use core::sync::atomic::Ordering::{Relaxed, Release, Acquire, SeqCst};
use core::borrow;
use core::fmt;
use core::cmp::Ordering;
use core::mem::{align_of_val, size_of_val};
use core::intrinsics::abort;
use core::mem;
use core::ops::Deref;
#[cfg(not(stage0))]
use core::ops::CoerceUnsized;
use core::ptr::{self, Shared};
#[cfg(not(stage0))]
use core::marker::Unsize;
use core::hash::{Hash, Hasher};
use core::{usize, isize};
use core::convert::From;
use heap::deallocate;

const MAX_REFCOUNT: usize = (isize::MAX) as usize;

/// An atomically reference counted wrapper for shared state.
///
/// # Examples
///
/// In this example, a large vector is shared between several threads.
/// With simple pipes, without `Arc`, a copy would have to be made for each
/// thread.
///
/// When you clone an `Arc<T>`, it will create another pointer to the data and
/// increase the reference counter.
///
/// ```
/// use std::sync::Arc;
/// use std::thread;
///
/// fn main() {
///     let numbers: Vec<_> = (0..100u32).collect();
///     let shared_numbers = Arc::new(numbers);
///
///     for _ in 0..10 {
///         let child_numbers = shared_numbers.clone();
///
///         thread::spawn(move || {
///             let local_numbers = &child_numbers[..];
///
///             // Work with the local numbers
///         });
///     }
/// }
/// ```
#[unsafe_no_drop_flag]
#[stable(feature = "rust1", since = "1.0.0")]
pub struct Arc<T: ?Sized> {
    // FIXME #12808: strange name to try to avoid interfering with
    // field accesses of the contained type via Deref
    _ptr: Shared<ArcInner<T>>,
}

#[stable(feature = "rust1", since = "1.0.0")]
unsafe impl<T: ?Sized + Sync + Send> Send for Arc<T> { }
#[stable(feature = "rust1", since = "1.0.0")]
unsafe impl<T: ?Sized + Sync + Send> Sync for Arc<T> { }

#[cfg(not(stage0))] // remove cfg after new snapshot
#[unstable(feature = "coerce_unsized", issue = "27732")]
impl<T: ?Sized + Unsize<U>, U: ?Sized> CoerceUnsized<Arc<U>> for Arc<T> {}

/// A weak pointer to an `Arc`.
///
/// Weak pointers will not keep the data inside of the `Arc` alive, and can be
/// used to break cycles between `Arc` pointers.
#[unsafe_no_drop_flag]
#[stable(feature = "arc_weak", since = "1.4.0")]
pub struct Weak<T: ?Sized> {
    // FIXME #12808: strange name to try to avoid interfering with
    // field accesses of the contained type via Deref
    _ptr: Shared<ArcInner<T>>,
}

#[stable(feature = "rust1", since = "1.0.0")]
unsafe impl<T: ?Sized + Sync + Send> Send for Weak<T> { }
#[stable(feature = "rust1", since = "1.0.0")]
unsafe impl<T: ?Sized + Sync + Send> Sync for Weak<T> { }

#[cfg(not(stage0))] // remove cfg after new snapshot
#[unstable(feature = "coerce_unsized", issue = "27732")]
impl<T: ?Sized + Unsize<U>, U: ?Sized> CoerceUnsized<Weak<U>> for Weak<T> {}

#[stable(feature = "rust1", since = "1.0.0")]
impl<T: ?Sized + fmt::Debug> fmt::Debug for Weak<T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "(Weak)")
    }
}

struct ArcInner<T: ?Sized> {
    strong: atomic::AtomicUsize,

    // the value usize::MAX acts as a sentinel for temporarily "locking" the
    // ability to upgrade weak pointers or downgrade strong ones; this is used
    // to avoid races in `make_mut` and `get_mut`.
    weak: atomic::AtomicUsize,

    data: T,
}

unsafe impl<T: ?Sized + Sync + Send> Send for ArcInner<T> {}
unsafe impl<T: ?Sized + Sync + Send> Sync for ArcInner<T> {}

impl<T> Arc<T> {
    /// Constructs a new `Arc<T>`.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::sync::Arc;
    ///
    /// let five = Arc::new(5);
    /// ```
    #[inline]
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn new(data: T) -> Arc<T> {
        // Start the weak pointer count as 1 which is the weak pointer that's
        // held by all the strong pointers (kinda), see std/rc.rs for more info
        let x: Box<_> = box ArcInner {
            strong: atomic::AtomicUsize::new(1),
            weak: atomic::AtomicUsize::new(1),
            data: data,
        };
        Arc { _ptr: unsafe { Shared::new(Box::into_raw(x)) } }
    }

    /// Unwraps the contained value if the `Arc<T>` has only one strong reference.
    /// This will succeed even if there are outstanding weak references.
    ///
    /// Otherwise, an `Err` is returned with the same `Arc<T>`.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::sync::Arc;
    ///
    /// let x = Arc::new(3);
    /// assert_eq!(Arc::try_unwrap(x), Ok(3));
    ///
    /// let x = Arc::new(4);
    /// let _y = x.clone();
    /// assert_eq!(Arc::try_unwrap(x), Err(Arc::new(4)));
    /// ```
    #[inline]
    #[stable(feature = "arc_unique", since = "1.4.0")]
    pub fn try_unwrap(this: Self) -> Result<T, Self> {
        // See `drop` for why all these atomics are like this
        if this.inner().strong.compare_and_swap(1, 0, Release) != 1 {
            return Err(this)
        }

        atomic::fence(Acquire);

        unsafe {
            let ptr = *this._ptr;
            let elem = ptr::read(&(*ptr).data);

            // Make a weak pointer to clean up the implicit strong-weak reference
            let _weak = Weak { _ptr: this._ptr };
            mem::forget(this);

            Ok(elem)
        }
    }
}

impl<T: ?Sized> Arc<T> {
    /// Downgrades the `Arc<T>` to a `Weak<T>` reference.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::sync::Arc;
    ///
    /// let five = Arc::new(5);
    ///
    /// let weak_five = Arc::downgrade(&five);
    /// ```
    #[stable(feature = "arc_weak", since = "1.4.0")]
    pub fn downgrade(this: &Self) -> Weak<T> {
        loop {
            // This Relaxed is OK because we're checking the value in the CAS
            // below.
            let cur = this.inner().weak.load(Relaxed);

            // check if the weak counter is currently "locked"; if so, spin.
            if cur == usize::MAX {
                continue
            }

            // NOTE: this code currently ignores the possibility of overflow
            // into usize::MAX; in general both Rc and Arc need to be adjusted
            // to deal with overflow.

            // Unlike with Clone(), we need this to be an Acquire read to
            // synchronize with the write coming from `is_unique`, so that the
            // events prior to that write happen before this read.
            if this.inner().weak.compare_and_swap(cur, cur + 1, Acquire) == cur {
                return Weak { _ptr: this._ptr }
            }
        }
    }

    /// Get the number of weak references to this value.
    #[inline]
    #[unstable(feature = "arc_counts", reason = "not clearly useful, and racy",
               issue = "28356")]
    pub fn weak_count(this: &Self) -> usize {
        this.inner().weak.load(SeqCst) - 1
    }

    /// Get the number of strong references to this value.
    #[inline]
    #[unstable(feature = "arc_counts", reason = "not clearly useful, and racy",
               issue = "28356")]
    pub fn strong_count(this: &Self) -> usize {
        this.inner().strong.load(SeqCst)
    }

    #[inline]
    fn inner(&self) -> &ArcInner<T> {
        // This unsafety is ok because while this arc is alive we're guaranteed
        // that the inner pointer is valid. Furthermore, we know that the
        // `ArcInner` structure itself is `Sync` because the inner data is
        // `Sync` as well, so we're ok loaning out an immutable pointer to these
        // contents.
        unsafe { &**self._ptr }
    }

    // Non-inlined part of `drop`.
    #[inline(never)]
    unsafe fn drop_slow(&mut self) {
        let ptr = *self._ptr;

        // Destroy the data at this time, even though we may not free the box
        // allocation itself (there may still be weak pointers lying around).
        ptr::drop_in_place(&mut (*ptr).data);

        if self.inner().weak.fetch_sub(1, Release) == 1 {
            atomic::fence(Acquire);
            deallocate(ptr as *mut u8, size_of_val(&*ptr), align_of_val(&*ptr))
        }
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<T: ?Sized> Clone for Arc<T> {
    /// Makes a clone of the `Arc<T>`.
    ///
    /// This increases the strong reference count.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::sync::Arc;
    ///
    /// let five = Arc::new(5);
    ///
    /// five.clone();
    /// ```
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
        let old_size = self.inner().strong.fetch_add(1, Relaxed);

        // However we need to guard against massive refcounts in case someone
        // is `mem::forget`ing Arcs. If we don't do this the count can overflow
        // and users will use-after free. We racily saturate to `isize::MAX` on
        // the assumption that there aren't ~2 billion threads incrementing
        // the reference count at once. This branch will never be taken in
        // any realistic program.
        //
        // We abort because such a program is incredibly degenerate, and we
        // don't care to support it.
        if old_size > MAX_REFCOUNT {
            unsafe {
                abort();
            }
        }

        Arc { _ptr: self._ptr }
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<T: ?Sized> Deref for Arc<T> {
    type Target = T;

    #[inline]
    fn deref(&self) -> &T {
        &self.inner().data
    }
}

impl<T: Clone> Arc<T> {
    #[unstable(feature = "arc_make_unique", reason = "renamed to Arc::make_mut",
               issue = "27718")]
    #[rustc_deprecated(since = "1.4.0", reason = "renamed to Arc::make_mut")]
    pub fn make_unique(this: &mut Self) -> &mut T {
        Arc::make_mut(this)
    }

    /// Make a mutable reference into the given `Arc<T>` by cloning the inner
    /// data if the `Arc<T>` doesn't have one strong reference and no weak
    /// references.
    ///
    /// This is also referred to as a copy-on-write.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::sync::Arc;
    ///
    /// let mut data = Arc::new(5);
    ///
    /// *Arc::make_mut(&mut data) += 1;         // Won't clone anything
    /// let mut other_data = data.clone();      // Won't clone inner data
    /// *Arc::make_mut(&mut data) += 1;         // Clones inner data
    /// *Arc::make_mut(&mut data) += 1;         // Won't clone anything
    /// *Arc::make_mut(&mut other_data) *= 2;   // Won't clone anything
    ///
    /// // Note: data and other_data now point to different numbers
    /// assert_eq!(*data, 8);
    /// assert_eq!(*other_data, 12);
    ///
    /// ```
    #[inline]
    #[stable(feature = "arc_unique", since = "1.4.0")]
    pub fn make_mut(this: &mut Self) -> &mut T {
        // Note that we hold both a strong reference and a weak reference.
        // Thus, releasing our strong reference only will not, by itself, cause
        // the memory to be deallocated.
        //
        // Use Acquire to ensure that we see any writes to `weak` that happen
        // before release writes (i.e., decrements) to `strong`. Since we hold a
        // weak count, there's no chance the ArcInner itself could be
        // deallocated.
        if this.inner().strong.compare_and_swap(1, 0, Acquire) != 1 {
            // Another srong pointer exists; clone
            *this = Arc::new((**this).clone());
        } else if this.inner().weak.load(Relaxed) != 1 {
            // Relaxed suffices in the above because this is fundamentally an
            // optimization: we are always racing with weak pointers being
            // dropped. Worst case, we end up allocated a new Arc unnecessarily.

            // We removed the last strong ref, but there are additional weak
            // refs remaining. We'll move the contents to a new Arc, and
            // invalidate the other weak refs.

            // Note that it is not possible for the read of `weak` to yield
            // usize::MAX (i.e., locked), since the weak count can only be
            // locked by a thread with a strong reference.

            // Materialize our own implicit weak pointer, so that it can clean
            // up the ArcInner as needed.
            let weak = Weak { _ptr: this._ptr };

            // mark the data itself as already deallocated
            unsafe {
                // there is no data race in the implicit write caused by `read`
                // here (due to zeroing) because data is no longer accessed by
                // other threads (due to there being no more strong refs at this
                // point).
                let mut swap = Arc::new(ptr::read(&(**weak._ptr).data));
                mem::swap(this, &mut swap);
                mem::forget(swap);
            }
        } else {
            // We were the sole reference of either kind; bump back up the
            // strong ref count.
            this.inner().strong.store(1, Release);
        }

        // As with `get_mut()`, the unsafety is ok because our reference was
        // either unique to begin with, or became one upon cloning the contents.
        unsafe {
            let inner = &mut **this._ptr;
            &mut inner.data
        }
    }
}

impl<T: ?Sized> Arc<T> {
    /// Returns a mutable reference to the contained value if the `Arc<T>` has
    /// one strong reference and no weak references.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::sync::Arc;
    ///
    /// let mut x = Arc::new(3);
    /// *Arc::get_mut(&mut x).unwrap() = 4;
    /// assert_eq!(*x, 4);
    ///
    /// let _y = x.clone();
    /// assert!(Arc::get_mut(&mut x).is_none());
    /// ```
    #[inline]
    #[stable(feature = "arc_unique", since = "1.4.0")]
    pub fn get_mut(this: &mut Self) -> Option<&mut T> {
        if this.is_unique() {
            // This unsafety is ok because we're guaranteed that the pointer
            // returned is the *only* pointer that will ever be returned to T. Our
            // reference count is guaranteed to be 1 at this point, and we required
            // the Arc itself to be `mut`, so we're returning the only possible
            // reference to the inner data.
            unsafe {
                let inner = &mut **this._ptr;
                Some(&mut inner.data)
            }
        } else {
            None
        }
    }

    /// Determine whether this is the unique reference (including weak refs) to
    /// the underlying data.
    ///
    /// Note that this requires locking the weak ref count.
    fn is_unique(&mut self) -> bool {
        // lock the weak pointer count if we appear to be the sole weak pointer
        // holder.
        //
        // The acquire label here ensures a happens-before relationship with any
        // writes to `strong` prior to decrements of the `weak` count (via drop,
        // which uses Release).
        if self.inner().weak.compare_and_swap(1, usize::MAX, Acquire) == 1 {
            // Due to the previous acquire read, this will observe any writes to
            // `strong` that were due to upgrading weak pointers; only strong
            // clones remain, which require that the strong count is > 1 anyway.
            let unique = self.inner().strong.load(Relaxed) == 1;

            // The release write here synchronizes with a read in `downgrade`,
            // effectively preventing the above read of `strong` from happening
            // after the write.
            self.inner().weak.store(1, Release); // release the lock
            unique
        } else {
            false
        }
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<T: ?Sized> Drop for Arc<T> {
    /// Drops the `Arc<T>`.
    ///
    /// This will decrement the strong reference count. If the strong reference
    /// count becomes zero and the only other references are `Weak<T>` ones,
    /// `drop`s the inner value.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::sync::Arc;
    ///
    /// {
    ///     let five = Arc::new(5);
    ///
    ///     // stuff
    ///
    ///     drop(five); // explicit drop
    /// }
    /// {
    ///     let five = Arc::new(5);
    ///
    ///     // stuff
    ///
    /// } // implicit drop
    /// ```
    #[unsafe_destructor_blind_to_params]
    #[inline]
    fn drop(&mut self) {
        // This structure has #[unsafe_no_drop_flag], so this drop glue may run
        // more than once (but it is guaranteed to be zeroed after the first if
        // it's run more than once)
        let ptr = *self._ptr;
        // if ptr.is_null() { return }
        if ptr as *mut u8 as usize == 0 || ptr as *mut u8 as usize == mem::POST_DROP_USIZE {
            return
        }

        // Because `fetch_sub` is already atomic, we do not need to synchronize
        // with other threads unless we are going to delete the object. This
        // same logic applies to the below `fetch_sub` to the `weak` count.
        if self.inner().strong.fetch_sub(1, Release) != 1 {
            return
        }

        // This fence is needed to prevent reordering of use of the data and
        // deletion of the data.  Because it is marked `Release`, the decreasing
        // of the reference count synchronizes with this `Acquire` fence. This
        // means that use of the data happens before decreasing the reference
        // count, which happens before this fence, which happens before the
        // deletion of the data.
        //
        // As explained in the [Boost documentation][1],
        //
        // > It is important to enforce any possible access to the object in one
        // > thread (through an existing reference) to *happen before* deleting
        // > the object in a different thread. This is achieved by a "release"
        // > operation after dropping a reference (any access to the object
        // > through this reference must obviously happened before), and an
        // > "acquire" operation before deleting the object.
        //
        // [1]: (www.boost.org/doc/libs/1_55_0/doc/html/atomic/usage_examples.html)
        atomic::fence(Acquire);

        unsafe {
            self.drop_slow();
        }
    }
}

impl<T: ?Sized> Weak<T> {
    /// Upgrades a weak reference to a strong reference.
    ///
    /// Upgrades the `Weak<T>` reference to an `Arc<T>`, if possible.
    ///
    /// Returns `None` if there were no strong references and the data was
    /// destroyed.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::sync::Arc;
    ///
    /// let five = Arc::new(5);
    ///
    /// let weak_five = Arc::downgrade(&five);
    ///
    /// let strong_five: Option<Arc<_>> = weak_five.upgrade();
    /// ```
    #[stable(feature = "arc_weak", since = "1.4.0")]
    pub fn upgrade(&self) -> Option<Arc<T>> {
        // We use a CAS loop to increment the strong count instead of a
        // fetch_add because once the count hits 0 it must never be above 0.
        let inner = self.inner();
        loop {
            // Relaxed load because any write of 0 that we can observe
            // leaves the field in a permanently zero state (so a
            // "stale" read of 0 is fine), and any other value is
            // confirmed via the CAS below.
            let n = inner.strong.load(Relaxed);
            if n == 0 {
                return None
            }

            // Relaxed is valid for the same reason it is on Arc's Clone impl
            let old = inner.strong.compare_and_swap(n, n + 1, Relaxed);
            if old == n {
                return Some(Arc { _ptr: self._ptr })
            }
        }
    }

    #[inline]
    fn inner(&self) -> &ArcInner<T> {
        // See comments above for why this is "safe"
        unsafe { &**self._ptr }
    }
}

#[stable(feature = "arc_weak", since = "1.4.0")]
impl<T: ?Sized> Clone for Weak<T> {
    /// Makes a clone of the `Weak<T>`.
    ///
    /// This increases the weak reference count.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::sync::Arc;
    ///
    /// let weak_five = Arc::downgrade(&Arc::new(5));
    ///
    /// weak_five.clone();
    /// ```
    #[inline]
    fn clone(&self) -> Weak<T> {
        // See comments in Arc::clone() for why this is relaxed.  This can use a
        // fetch_add (ignoring the lock) because the weak count is only locked
        // where are *no other* weak pointers in existence. (So we can't be
        // running this code in that case).
        let old_size = self.inner().weak.fetch_add(1, Relaxed);

        // See comments in Arc::clone() for why we do this (for mem::forget).
        if old_size > MAX_REFCOUNT {
            unsafe {
                abort();
            }
        }

        return Weak { _ptr: self._ptr }
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<T: ?Sized> Drop for Weak<T> {
    /// Drops the `Weak<T>`.
    ///
    /// This will decrement the weak reference count.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::sync::Arc;
    ///
    /// {
    ///     let five = Arc::new(5);
    ///     let weak_five = Arc::downgrade(&five);
    ///
    ///     // stuff
    ///
    ///     drop(weak_five); // explicit drop
    /// }
    /// {
    ///     let five = Arc::new(5);
    ///     let weak_five = Arc::downgrade(&five);
    ///
    ///     // stuff
    ///
    /// } // implicit drop
    /// ```
    fn drop(&mut self) {
        let ptr = *self._ptr;

        // see comments above for why this check is here
        if ptr as *mut u8 as usize == 0 || ptr as *mut u8 as usize == mem::POST_DROP_USIZE {
            return
        }

        // If we find out that we were the last weak pointer, then its time to
        // deallocate the data entirely. See the discussion in Arc::drop() about
        // the memory orderings
        //
        // It's not necessary to check for the locked state here, because the
        // weak count can only be locked if there was precisely one weak ref,
        // meaning that drop could only subsequently run ON that remaining weak
        // ref, which can only happen after the lock is released.
        if self.inner().weak.fetch_sub(1, Release) == 1 {
            atomic::fence(Acquire);
            unsafe { deallocate(ptr as *mut u8, size_of_val(&*ptr), align_of_val(&*ptr)) }
        }
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<T: ?Sized + PartialEq> PartialEq for Arc<T> {
    /// Equality for two `Arc<T>`s.
    ///
    /// Two `Arc<T>`s are equal if their inner value are equal.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::sync::Arc;
    ///
    /// let five = Arc::new(5);
    ///
    /// five == Arc::new(5);
    /// ```
    fn eq(&self, other: &Arc<T>) -> bool {
        *(*self) == *(*other)
    }

    /// Inequality for two `Arc<T>`s.
    ///
    /// Two `Arc<T>`s are unequal if their inner value are unequal.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::sync::Arc;
    ///
    /// let five = Arc::new(5);
    ///
    /// five != Arc::new(5);
    /// ```
    fn ne(&self, other: &Arc<T>) -> bool {
        *(*self) != *(*other)
    }
}
#[stable(feature = "rust1", since = "1.0.0")]
impl<T: ?Sized + PartialOrd> PartialOrd for Arc<T> {
    /// Partial comparison for two `Arc<T>`s.
    ///
    /// The two are compared by calling `partial_cmp()` on their inner values.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::sync::Arc;
    ///
    /// let five = Arc::new(5);
    ///
    /// five.partial_cmp(&Arc::new(5));
    /// ```
    fn partial_cmp(&self, other: &Arc<T>) -> Option<Ordering> {
        (**self).partial_cmp(&**other)
    }

    /// Less-than comparison for two `Arc<T>`s.
    ///
    /// The two are compared by calling `<` on their inner values.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::sync::Arc;
    ///
    /// let five = Arc::new(5);
    ///
    /// five < Arc::new(5);
    /// ```
    fn lt(&self, other: &Arc<T>) -> bool {
        *(*self) < *(*other)
    }

    /// 'Less-than or equal to' comparison for two `Arc<T>`s.
    ///
    /// The two are compared by calling `<=` on their inner values.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::sync::Arc;
    ///
    /// let five = Arc::new(5);
    ///
    /// five <= Arc::new(5);
    /// ```
    fn le(&self, other: &Arc<T>) -> bool {
        *(*self) <= *(*other)
    }

    /// Greater-than comparison for two `Arc<T>`s.
    ///
    /// The two are compared by calling `>` on their inner values.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::sync::Arc;
    ///
    /// let five = Arc::new(5);
    ///
    /// five > Arc::new(5);
    /// ```
    fn gt(&self, other: &Arc<T>) -> bool {
        *(*self) > *(*other)
    }

    /// 'Greater-than or equal to' comparison for two `Arc<T>`s.
    ///
    /// The two are compared by calling `>=` on their inner values.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::sync::Arc;
    ///
    /// let five = Arc::new(5);
    ///
    /// five >= Arc::new(5);
    /// ```
    fn ge(&self, other: &Arc<T>) -> bool {
        *(*self) >= *(*other)
    }
}
#[stable(feature = "rust1", since = "1.0.0")]
impl<T: ?Sized + Ord> Ord for Arc<T> {
    fn cmp(&self, other: &Arc<T>) -> Ordering {
        (**self).cmp(&**other)
    }
}
#[stable(feature = "rust1", since = "1.0.0")]
impl<T: ?Sized + Eq> Eq for Arc<T> {}

#[stable(feature = "rust1", since = "1.0.0")]
impl<T: ?Sized + fmt::Display> fmt::Display for Arc<T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        fmt::Display::fmt(&**self, f)
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<T: ?Sized + fmt::Debug> fmt::Debug for Arc<T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        fmt::Debug::fmt(&**self, f)
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<T> fmt::Pointer for Arc<T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        fmt::Pointer::fmt(&*self._ptr, f)
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<T: Default> Default for Arc<T> {
    fn default() -> Arc<T> {
        Arc::new(Default::default())
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<T: ?Sized + Hash> Hash for Arc<T> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        (**self).hash(state)
    }
}

#[stable(feature = "from_for_ptrs", since = "1.6.0")]
impl<T> From<T> for Arc<T> {
    fn from(t: T) -> Self {
        Arc::new(t)
    }
}

#[cfg(test)]
mod tests {
    use std::clone::Clone;
    use std::sync::mpsc::channel;
    use std::mem::drop;
    use std::ops::Drop;
    use std::option::Option;
    use std::option::Option::{Some, None};
    use std::sync::atomic;
    use std::sync::atomic::Ordering::{Acquire, SeqCst};
    use std::thread;
    use std::vec::Vec;
    use super::{Arc, Weak};
    use std::sync::Mutex;
    use std::convert::From;

    struct Canary(*mut atomic::AtomicUsize);

    impl Drop for Canary
    {
        fn drop(&mut self) {
            unsafe {
                match *self {
                    Canary(c) => {
                        (*c).fetch_add(1, SeqCst);
                    }
                }
            }
        }
    }

    #[test]
    fn manually_share_arc() {
        let v = vec!(1, 2, 3, 4, 5, 6, 7, 8, 9, 10);
        let arc_v = Arc::new(v);

        let (tx, rx) = channel();

        let _t = thread::spawn(move || {
            let arc_v: Arc<Vec<i32>> = rx.recv().unwrap();
            assert_eq!((*arc_v)[3], 4);
        });

        tx.send(arc_v.clone()).unwrap();

        assert_eq!((*arc_v)[2], 3);
        assert_eq!((*arc_v)[4], 5);
    }

    #[test]
    fn test_arc_get_mut() {
        let mut x = Arc::new(3);
        *Arc::get_mut(&mut x).unwrap() = 4;
        assert_eq!(*x, 4);
        let y = x.clone();
        assert!(Arc::get_mut(&mut x).is_none());
        drop(y);
        assert!(Arc::get_mut(&mut x).is_some());
        let _w = Arc::downgrade(&x);
        assert!(Arc::get_mut(&mut x).is_none());
    }

    #[test]
    fn try_unwrap() {
        let x = Arc::new(3);
        assert_eq!(Arc::try_unwrap(x), Ok(3));
        let x = Arc::new(4);
        let _y = x.clone();
        assert_eq!(Arc::try_unwrap(x), Err(Arc::new(4)));
        let x = Arc::new(5);
        let _w = Arc::downgrade(&x);
        assert_eq!(Arc::try_unwrap(x), Ok(5));
    }

    #[test]
    fn test_cowarc_clone_make_mut() {
        let mut cow0 = Arc::new(75);
        let mut cow1 = cow0.clone();
        let mut cow2 = cow1.clone();

        assert!(75 == *Arc::make_mut(&mut cow0));
        assert!(75 == *Arc::make_mut(&mut cow1));
        assert!(75 == *Arc::make_mut(&mut cow2));

        *Arc::make_mut(&mut cow0) += 1;
        *Arc::make_mut(&mut cow1) += 2;
        *Arc::make_mut(&mut cow2) += 3;

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
        let mut cow0 = Arc::new(75);
        let cow1 = cow0.clone();
        let cow2 = cow1.clone();

        assert!(75 == *cow0);
        assert!(75 == *cow1);
        assert!(75 == *cow2);

        *Arc::make_mut(&mut cow0) += 1;
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
    fn test_cowarc_clone_weak() {
        let mut cow0 = Arc::new(75);
        let cow1_weak = Arc::downgrade(&cow0);

        assert!(75 == *cow0);
        assert!(75 == *cow1_weak.upgrade().unwrap());

        *Arc::make_mut(&mut cow0) += 1;

        assert!(76 == *cow0);
        assert!(cow1_weak.upgrade().is_none());
    }

    #[test]
    fn test_live() {
        let x = Arc::new(5);
        let y = Arc::downgrade(&x);
        assert!(y.upgrade().is_some());
    }

    #[test]
    fn test_dead() {
        let x = Arc::new(5);
        let y = Arc::downgrade(&x);
        drop(x);
        assert!(y.upgrade().is_none());
    }

    #[test]
    fn weak_self_cyclic() {
        struct Cycle {
            x: Mutex<Option<Weak<Cycle>>>,
        }

        let a = Arc::new(Cycle { x: Mutex::new(None) });
        let b = Arc::downgrade(&a.clone());
        *a.x.lock().unwrap() = Some(b);

        // hopefully we don't double-free (or leak)...
    }

    #[test]
    fn drop_arc() {
        let mut canary = atomic::AtomicUsize::new(0);
        let x = Arc::new(Canary(&mut canary as *mut atomic::AtomicUsize));
        drop(x);
        assert!(canary.load(Acquire) == 1);
    }

    #[test]
    fn drop_arc_weak() {
        let mut canary = atomic::AtomicUsize::new(0);
        let arc = Arc::new(Canary(&mut canary as *mut atomic::AtomicUsize));
        let arc_weak = Arc::downgrade(&arc);
        assert!(canary.load(Acquire) == 0);
        drop(arc);
        assert!(canary.load(Acquire) == 1);
        drop(arc_weak);
    }

    #[test]
    fn test_strong_count() {
        let a = Arc::new(0u32);
        assert!(Arc::strong_count(&a) == 1);
        let w = Arc::downgrade(&a);
        assert!(Arc::strong_count(&a) == 1);
        let b = w.upgrade().expect("");
        assert!(Arc::strong_count(&b) == 2);
        assert!(Arc::strong_count(&a) == 2);
        drop(w);
        drop(a);
        assert!(Arc::strong_count(&b) == 1);
        let c = b.clone();
        assert!(Arc::strong_count(&b) == 2);
        assert!(Arc::strong_count(&c) == 2);
    }

    #[test]
    fn test_weak_count() {
        let a = Arc::new(0u32);
        assert!(Arc::strong_count(&a) == 1);
        assert!(Arc::weak_count(&a) == 0);
        let w = Arc::downgrade(&a);
        assert!(Arc::strong_count(&a) == 1);
        assert!(Arc::weak_count(&a) == 1);
        let x = w.clone();
        assert!(Arc::weak_count(&a) == 2);
        drop(w);
        drop(x);
        assert!(Arc::strong_count(&a) == 1);
        assert!(Arc::weak_count(&a) == 0);
        let c = a.clone();
        assert!(Arc::strong_count(&a) == 2);
        assert!(Arc::weak_count(&a) == 0);
        let d = Arc::downgrade(&c);
        assert!(Arc::weak_count(&c) == 1);
        assert!(Arc::strong_count(&c) == 2);

        drop(a);
        drop(c);
        drop(d);
    }

    #[test]
    fn show_arc() {
        let a = Arc::new(5u32);
        assert_eq!(format!("{:?}", a), "5");
    }

    // Make sure deriving works with Arc<T>
    #[derive(Eq, Ord, PartialEq, PartialOrd, Clone, Debug, Default)]
    struct Foo {
        inner: Arc<i32>,
    }

    #[test]
    fn test_unsized() {
        let x: Arc<[i32]> = Arc::new([1, 2, 3]);
        assert_eq!(format!("{:?}", x), "[1, 2, 3]");
        let y = Arc::downgrade(&x.clone());
        drop(x);
        assert!(y.upgrade().is_none());
    }

    #[test]
    fn test_from_owned() {
        let foo = 123;
        let foo_arc = Arc::from(foo);
        assert!(123 == *foo_arc);
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<T: ?Sized> borrow::Borrow<T> for Arc<T> {
    fn borrow(&self) -> &T {
        &**self
    }
}

#[stable(since = "1.5.0", feature = "smart_ptr_as_ref")]
impl<T: ?Sized> AsRef<T> for Arc<T> {
    fn as_ref(&self) -> &T {
        &**self
    }
}
