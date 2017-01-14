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

//! Thread-safe reference-counting pointers.
//!
//! See the [`Arc<T>`][arc] documentation for more details.
//!
//! [arc]: struct.Arc.html

use boxed::Box;

use core::sync::atomic;
use core::sync::atomic::Ordering::{Acquire, Relaxed, Release, SeqCst};
use core::borrow;
use core::fmt;
use core::cmp::Ordering;
use core::mem::{align_of_val, size_of_val};
use core::intrinsics::abort;
use core::mem;
use core::mem::uninitialized;
use core::ops::Deref;
use core::ops::CoerceUnsized;
use core::ptr::{self, Shared};
use core::marker::Unsize;
use core::hash::{Hash, Hasher};
use core::{isize, usize};
use core::convert::From;
use heap::deallocate;

/// A soft limit on the amount of references that may be made to an `Arc`.
///
/// Going above this limit will abort your program (although not
/// necessarily) at _exactly_ `MAX_REFCOUNT + 1` references.
const MAX_REFCOUNT: usize = (isize::MAX) as usize;

/// A thread-safe reference-counting pointer.
///
/// The type `Arc<T>` provides shared ownership of a value of type `T`,
/// allocated in the heap. Invoking [`clone`][clone] on `Arc` produces
/// a new pointer to the same value in the heap. When the last `Arc`
/// pointer to a given value is destroyed, the pointed-to value is
/// also destroyed.
///
/// Shared references in Rust disallow mutation by default, and `Arc` is no
/// exception. If you need to mutate through an `Arc`, use [`Mutex`][mutex],
/// [`RwLock`][rwlock], or one of the [`Atomic`][atomic] types.
///
/// `Arc` uses atomic operations for reference counting, so `Arc`s can be
/// sent between threads. In other words, `Arc<T>` implements [`Send`]
/// as long as `T` implements [`Send`] and [`Sync`][sync]. The disadvantage is
/// that atomic operations are more expensive than ordinary memory accesses.
/// If you are not sharing reference-counted values between threads, consider
/// using [`rc::Rc`] for lower overhead. [`Rc`] is a safe default, because
/// the compiler will catch any attempt to send an [`Rc`] between threads.
/// However, a library might choose `Arc` in order to give library consumers
/// more flexibility.
///
/// The [`downgrade`][downgrade] method can be used to create a non-owning
/// [`Weak`][weak] pointer. A [`Weak`][weak] pointer can be [`upgrade`][upgrade]d
/// to an `Arc`, but this will return [`None`] if the value has already been
/// dropped.
///
/// A cycle between `Arc` pointers will never be deallocated. For this reason,
/// [`Weak`][weak] is used to break cycles. For example, a tree could have
/// strong `Arc` pointers from parent nodes to children, and [`Weak`][weak]
/// pointers from children back to their parents.
///
/// `Arc<T>` automatically dereferences to `T` (via the [`Deref`][deref] trait),
/// so you can call `T`'s methods on a value of type `Arc<T>`. To avoid name
/// clashes with `T`'s methods, the methods of `Arc<T>` itself are [associated
/// functions][assoc], called using function-like syntax:
///
/// ```
/// use std::sync::Arc;
/// let my_arc = Arc::new(());
///
/// Arc::downgrade(&my_arc);
/// ```
///
/// [`Weak<T>`][weak] does not auto-dereference to `T`, because the value may have
/// already been destroyed.
///
/// [arc]: struct.Arc.html
/// [weak]: struct.Weak.html
/// [`Rc`]: ../../std/rc/struct.Rc.html
/// [clone]: ../../std/clone/trait.Clone.html#tymethod.clone
/// [mutex]: ../../std/sync/struct.Mutex.html
/// [rwlock]: ../../std/sync/struct.RwLock.html
/// [atomic]: ../../std/sync/atomic/index.html
/// [`Send`]: ../../std/marker/trait.Send.html
/// [sync]: ../../std/marker/trait.Sync.html
/// [deref]: ../../std/ops/trait.Deref.html
/// [downgrade]: struct.Arc.html#method.downgrade
/// [upgrade]: struct.Weak.html#method.upgrade
/// [`None`]: ../../std/option/enum.Option.html#variant.None
/// [assoc]: ../../book/method-syntax.html#associated-functions
///
/// # Examples
///
/// Sharing some immutable data between threads:
///
// Note that we **do not** run these tests here. The windows builders get super
// unhappy if a thread outlives the main thread and then exits at the same time
// (something deadlocks) so we just avoid this entirely by not running these
// tests.
/// ```no_run
/// use std::sync::Arc;
/// use std::thread;
///
/// let five = Arc::new(5);
///
/// for _ in 0..10 {
///     let five = five.clone();
///
///     thread::spawn(move || {
///         println!("{:?}", five);
///     });
/// }
/// ```
///
/// Sharing a mutable [`AtomicUsize`]:
///
/// [`AtomicUsize`]: ../../std/sync/atomic/struct.AtomicUsize.html
///
/// ```no_run
/// use std::sync::Arc;
/// use std::sync::atomic::{AtomicUsize, Ordering};
/// use std::thread;
///
/// let val = Arc::new(AtomicUsize::new(5));
///
/// for _ in 0..10 {
///     let val = val.clone();
///
///     thread::spawn(move || {
///         let v = val.fetch_add(1, Ordering::SeqCst);
///         println!("{:?}", v);
///     });
/// }
/// ```
///
/// See the [`rc` documentation][rc_examples] for more examples of reference
/// counting in general.
///
/// [rc_examples]: ../../std/rc/index.html#examples
#[stable(feature = "rust1", since = "1.0.0")]
pub struct Arc<T: ?Sized> {
    ptr: Shared<ArcInner<T>>,
}

#[stable(feature = "rust1", since = "1.0.0")]
unsafe impl<T: ?Sized + Sync + Send> Send for Arc<T> {}
#[stable(feature = "rust1", since = "1.0.0")]
unsafe impl<T: ?Sized + Sync + Send> Sync for Arc<T> {}

#[unstable(feature = "coerce_unsized", issue = "27732")]
impl<T: ?Sized + Unsize<U>, U: ?Sized> CoerceUnsized<Arc<U>> for Arc<T> {}

/// A weak version of [`Arc`][arc].
///
/// `Weak` pointers do not count towards determining if the inner value
/// should be dropped.
///
/// The typical way to obtain a `Weak` pointer is to call
/// [`Arc::downgrade`][downgrade].
///
/// See the [`Arc`][arc] documentation for more details.
///
/// [arc]: struct.Arc.html
/// [downgrade]: struct.Arc.html#method.downgrade
#[stable(feature = "arc_weak", since = "1.4.0")]
pub struct Weak<T: ?Sized> {
    ptr: Shared<ArcInner<T>>,
}

#[stable(feature = "arc_weak", since = "1.4.0")]
unsafe impl<T: ?Sized + Sync + Send> Send for Weak<T> {}
#[stable(feature = "arc_weak", since = "1.4.0")]
unsafe impl<T: ?Sized + Sync + Send> Sync for Weak<T> {}

#[unstable(feature = "coerce_unsized", issue = "27732")]
impl<T: ?Sized + Unsize<U>, U: ?Sized> CoerceUnsized<Weak<U>> for Weak<T> {}

#[stable(feature = "arc_weak", since = "1.4.0")]
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
        Arc { ptr: unsafe { Shared::new(Box::into_raw(x)) } }
    }

    /// Returns the contained value, if the `Arc` has exactly one strong reference.
    ///
    /// Otherwise, an [`Err`][result] is returned with the same `Arc` that was
    /// passed in.
    ///
    /// This will succeed even if there are outstanding weak references.
    ///
    /// [result]: ../../std/result/enum.Result.html
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
    /// assert_eq!(*Arc::try_unwrap(x).unwrap_err(), 4);
    /// ```
    #[inline]
    #[stable(feature = "arc_unique", since = "1.4.0")]
    pub fn try_unwrap(this: Self) -> Result<T, Self> {
        // See `drop` for why all these atomics are like this
        if this.inner().strong.compare_exchange(1, 0, Release, Relaxed).is_err() {
            return Err(this);
        }

        atomic::fence(Acquire);

        unsafe {
            let ptr = *this.ptr;
            let elem = ptr::read(&(*ptr).data);

            // Make a weak pointer to clean up the implicit strong-weak reference
            let _weak = Weak { ptr: this.ptr };
            mem::forget(this);

            Ok(elem)
        }
    }

    /// Consumes the `Arc`, returning the wrapped pointer.
    ///
    /// To avoid a memory leak the pointer must be converted back to an `Arc` using
    /// [`Arc::from_raw`][from_raw].
    ///
    /// [from_raw]: struct.Arc.html#method.from_raw
    ///
    /// # Examples
    ///
    /// ```
    /// #![feature(rc_raw)]
    ///
    /// use std::sync::Arc;
    ///
    /// let x = Arc::new(10);
    /// let x_ptr = Arc::into_raw(x);
    /// assert_eq!(unsafe { *x_ptr }, 10);
    /// ```
    #[unstable(feature = "rc_raw", issue = "37197")]
    pub fn into_raw(this: Self) -> *mut T {
        let ptr = unsafe { &mut (**this.ptr).data as *mut _ };
        mem::forget(this);
        ptr
    }

    /// Constructs an `Arc` from a raw pointer.
    ///
    /// The raw pointer must have been previously returned by a call to a
    /// [`Arc::into_raw`][into_raw].
    ///
    /// This function is unsafe because improper use may lead to memory problems. For example, a
    /// double-free may occur if the function is called twice on the same raw pointer.
    ///
    /// [into_raw]: struct.Arc.html#method.into_raw
    ///
    /// # Examples
    ///
    /// ```
    /// #![feature(rc_raw)]
    ///
    /// use std::sync::Arc;
    ///
    /// let x = Arc::new(10);
    /// let x_ptr = Arc::into_raw(x);
    ///
    /// unsafe {
    ///     // Convert back to an `Arc` to prevent leak.
    ///     let x = Arc::from_raw(x_ptr);
    ///     assert_eq!(*x, 10);
    ///
    ///     // Further calls to `Arc::from_raw(x_ptr)` would be memory unsafe.
    /// }
    ///
    /// // The memory was freed when `x` went out of scope above, so `x_ptr` is now dangling!
    /// ```
    #[unstable(feature = "rc_raw", issue = "37197")]
    pub unsafe fn from_raw(ptr: *mut T) -> Self {
        // To find the corresponding pointer to the `ArcInner` we need to subtract the offset of the
        // `data` field from the pointer.
        Arc { ptr: Shared::new((ptr as *mut u8).offset(-offset_of!(ArcInner<T>, data)) as *mut _) }
    }
}

impl<T: ?Sized> Arc<T> {
    /// Creates a new [`Weak`][weak] pointer to this value.
    ///
    /// [weak]: struct.Weak.html
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
        // This Relaxed is OK because we're checking the value in the CAS
        // below.
        let mut cur = this.inner().weak.load(Relaxed);

        loop {
            // check if the weak counter is currently "locked"; if so, spin.
            if cur == usize::MAX {
                cur = this.inner().weak.load(Relaxed);
                continue;
            }

            // NOTE: this code currently ignores the possibility of overflow
            // into usize::MAX; in general both Rc and Arc need to be adjusted
            // to deal with overflow.

            // Unlike with Clone(), we need this to be an Acquire read to
            // synchronize with the write coming from `is_unique`, so that the
            // events prior to that write happen before this read.
            match this.inner().weak.compare_exchange_weak(cur, cur + 1, Acquire, Relaxed) {
                Ok(_) => return Weak { ptr: this.ptr },
                Err(old) => cur = old,
            }
        }
    }

    /// Gets the number of [`Weak`][weak] pointers to this value.
    ///
    /// [weak]: struct.Weak.html
    ///
    /// # Safety
    ///
    /// This method by itself is safe, but using it correctly requires extra care.
    /// Another thread can change the weak count at any time,
    /// including potentially between calling this method and acting on the result.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::sync::Arc;
    ///
    /// let five = Arc::new(5);
    /// let _weak_five = Arc::downgrade(&five);
    ///
    /// // This assertion is deterministic because we haven't shared
    /// // the `Arc` or `Weak` between threads.
    /// assert_eq!(1, Arc::weak_count(&five));
    /// ```
    #[inline]
    #[stable(feature = "arc_counts", since = "1.15.0")]
    pub fn weak_count(this: &Self) -> usize {
        this.inner().weak.load(SeqCst) - 1
    }

    /// Gets the number of strong (`Arc`) pointers to this value.
    ///
    /// # Safety
    ///
    /// This method by itself is safe, but using it correctly requires extra care.
    /// Another thread can change the strong count at any time,
    /// including potentially between calling this method and acting on the result.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::sync::Arc;
    ///
    /// let five = Arc::new(5);
    /// let _also_five = five.clone();
    ///
    /// // This assertion is deterministic because we haven't shared
    /// // the `Arc` between threads.
    /// assert_eq!(2, Arc::strong_count(&five));
    /// ```
    #[inline]
    #[stable(feature = "arc_counts", since = "1.15.0")]
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
        unsafe { &**self.ptr }
    }

    // Non-inlined part of `drop`.
    #[inline(never)]
    unsafe fn drop_slow(&mut self) {
        let ptr = *self.ptr;

        // Destroy the data at this time, even though we may not free the box
        // allocation itself (there may still be weak pointers lying around).
        ptr::drop_in_place(&mut (*ptr).data);

        if self.inner().weak.fetch_sub(1, Release) == 1 {
            atomic::fence(Acquire);
            deallocate(ptr as *mut u8, size_of_val(&*ptr), align_of_val(&*ptr))
        }
    }

    #[inline]
    #[unstable(feature = "ptr_eq",
               reason = "newly added",
               issue = "36497")]
    /// Returns true if the two `Arc`s point to the same value (not
    /// just values that compare as equal).
    ///
    /// # Examples
    ///
    /// ```
    /// #![feature(ptr_eq)]
    ///
    /// use std::sync::Arc;
    ///
    /// let five = Arc::new(5);
    /// let same_five = five.clone();
    /// let other_five = Arc::new(5);
    ///
    /// assert!(Arc::ptr_eq(&five, &same_five));
    /// assert!(!Arc::ptr_eq(&five, &other_five));
    /// ```
    pub fn ptr_eq(this: &Self, other: &Self) -> bool {
        let this_ptr: *const ArcInner<T> = *this.ptr;
        let other_ptr: *const ArcInner<T> = *other.ptr;
        this_ptr == other_ptr
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<T: ?Sized> Clone for Arc<T> {
    /// Makes a clone of the `Arc` pointer.
    ///
    /// This creates another pointer to the same inner value, increasing the
    /// strong reference count.
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

        Arc { ptr: self.ptr }
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
    /// Makes a mutable reference into the given `Arc`.
    ///
    /// If there are other `Arc` or [`Weak`][weak] pointers to the same value,
    /// then `make_mut` will invoke [`clone`][clone] on the inner value to
    /// ensure unique ownership. This is also referred to as clone-on-write.
    ///
    /// See also [`get_mut`][get_mut], which will fail rather than cloning.
    ///
    /// [weak]: struct.Weak.html
    /// [clone]: ../../std/clone/trait.Clone.html#tymethod.clone
    /// [get_mut]: struct.Arc.html#method.get_mut
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
    /// // Now `data` and `other_data` point to different values.
    /// assert_eq!(*data, 8);
    /// assert_eq!(*other_data, 12);
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
        if this.inner().strong.compare_exchange(1, 0, Acquire, Relaxed).is_err() {
            // Another strong pointer exists; clone
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
            let weak = Weak { ptr: this.ptr };

            // mark the data itself as already deallocated
            unsafe {
                // there is no data race in the implicit write caused by `read`
                // here (due to zeroing) because data is no longer accessed by
                // other threads (due to there being no more strong refs at this
                // point).
                let mut swap = Arc::new(ptr::read(&(**weak.ptr).data));
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
            let inner = &mut **this.ptr;
            &mut inner.data
        }
    }
}

impl<T: ?Sized> Arc<T> {
    /// Returns a mutable reference to the inner value, if there are
    /// no other `Arc` or [`Weak`][weak] pointers to the same value.
    ///
    /// Returns [`None`][option] otherwise, because it is not safe to
    /// mutate a shared value.
    ///
    /// See also [`make_mut`][make_mut], which will [`clone`][clone]
    /// the inner value when it's shared.
    ///
    /// [weak]: struct.Weak.html
    /// [option]: ../../std/option/enum.Option.html
    /// [make_mut]: struct.Arc.html#method.make_mut
    /// [clone]: ../../std/clone/trait.Clone.html#tymethod.clone
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
                let inner = &mut **this.ptr;
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
        if self.inner().weak.compare_exchange(1, usize::MAX, Acquire, Relaxed).is_ok() {
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
unsafe impl<#[may_dangle] T: ?Sized> Drop for Arc<T> {
    /// Drops the `Arc`.
    ///
    /// This will decrement the strong reference count. If the strong reference
    /// count reaches zero then the only other references (if any) are
    /// [`Weak`][weak], so we `drop` the inner value.
    ///
    /// [weak]: struct.Weak.html
    ///
    /// # Examples
    ///
    /// ```
    /// use std::sync::Arc;
    ///
    /// struct Foo;
    ///
    /// impl Drop for Foo {
    ///     fn drop(&mut self) {
    ///         println!("dropped!");
    ///     }
    /// }
    ///
    /// let foo  = Arc::new(Foo);
    /// let foo2 = foo.clone();
    ///
    /// drop(foo);    // Doesn't print anything
    /// drop(foo2);   // Prints "dropped!"
    /// ```
    #[inline]
    fn drop(&mut self) {
        // Because `fetch_sub` is already atomic, we do not need to synchronize
        // with other threads unless we are going to delete the object. This
        // same logic applies to the below `fetch_sub` to the `weak` count.
        if self.inner().strong.fetch_sub(1, Release) != 1 {
            return;
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

impl<T> Weak<T> {
    /// Constructs a new `Weak<T>`, without an accompanying instance of `T`.
    ///
    /// This allocates memory for `T`, but does not initialize it. Calling
    /// [`upgrade`][upgrade] on the return value always gives
    /// [`None`][option].
    ///
    /// [upgrade]: struct.Weak.html#method.upgrade
    /// [option]: ../../std/option/enum.Option.html
    ///
    /// # Examples
    ///
    /// ```
    /// use std::sync::Weak;
    ///
    /// let empty: Weak<i64> = Weak::new();
    /// assert!(empty.upgrade().is_none());
    /// ```
    #[stable(feature = "downgraded_weak", since = "1.10.0")]
    pub fn new() -> Weak<T> {
        unsafe {
            Weak {
                ptr: Shared::new(Box::into_raw(box ArcInner {
                    strong: atomic::AtomicUsize::new(0),
                    weak: atomic::AtomicUsize::new(1),
                    data: uninitialized(),
                })),
            }
        }
    }
}

impl<T: ?Sized> Weak<T> {
    /// Upgrades the `Weak` pointer to an [`Arc`][arc], if possible.
    ///
    /// Returns [`None`][option] if the strong count has reached zero and the
    /// inner value was destroyed.
    ///
    /// [arc]: struct.Arc.html
    /// [option]: ../../std/option/enum.Option.html
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
    /// assert!(strong_five.is_some());
    ///
    /// // Destroy all strong pointers.
    /// drop(strong_five);
    /// drop(five);
    ///
    /// assert!(weak_five.upgrade().is_none());
    /// ```
    #[stable(feature = "arc_weak", since = "1.4.0")]
    pub fn upgrade(&self) -> Option<Arc<T>> {
        // We use a CAS loop to increment the strong count instead of a
        // fetch_add because once the count hits 0 it must never be above 0.
        let inner = self.inner();

        // Relaxed load because any write of 0 that we can observe
        // leaves the field in a permanently zero state (so a
        // "stale" read of 0 is fine), and any other value is
        // confirmed via the CAS below.
        let mut n = inner.strong.load(Relaxed);

        loop {
            if n == 0 {
                return None;
            }

            // See comments in `Arc::clone` for why we do this (for `mem::forget`).
            if n > MAX_REFCOUNT {
                unsafe {
                    abort();
                }
            }

            // Relaxed is valid for the same reason it is on Arc's Clone impl
            match inner.strong.compare_exchange_weak(n, n + 1, Relaxed, Relaxed) {
                Ok(_) => return Some(Arc { ptr: self.ptr }),
                Err(old) => n = old,
            }
        }
    }

    #[inline]
    fn inner(&self) -> &ArcInner<T> {
        // See comments above for why this is "safe"
        unsafe { &**self.ptr }
    }
}

#[stable(feature = "arc_weak", since = "1.4.0")]
impl<T: ?Sized> Clone for Weak<T> {
    /// Makes a clone of the `Weak` pointer.
    ///
    /// This creates another pointer to the same inner value, increasing the
    /// weak reference count.
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

        return Weak { ptr: self.ptr };
    }
}

#[stable(feature = "downgraded_weak", since = "1.10.0")]
impl<T> Default for Weak<T> {
    /// Constructs a new `Weak<T>`, without an accompanying instance of `T`.
    ///
    /// This allocates memory for `T`, but does not initialize it. Calling
    /// [`upgrade`][upgrade] on the return value always gives
    /// [`None`][option].
    ///
    /// [upgrade]: struct.Weak.html#method.upgrade
    /// [option]: ../../std/option/enum.Option.html
    ///
    /// # Examples
    ///
    /// ```
    /// use std::sync::Weak;
    ///
    /// let empty: Weak<i64> = Default::default();
    /// assert!(empty.upgrade().is_none());
    /// ```
    fn default() -> Weak<T> {
        Weak::new()
    }
}

#[stable(feature = "arc_weak", since = "1.4.0")]
impl<T: ?Sized> Drop for Weak<T> {
    /// Drops the `Weak` pointer.
    ///
    /// This will decrement the weak reference count.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::sync::Arc;
    ///
    /// struct Foo;
    ///
    /// impl Drop for Foo {
    ///     fn drop(&mut self) {
    ///         println!("dropped!");
    ///     }
    /// }
    ///
    /// let foo = Arc::new(Foo);
    /// let weak_foo = Arc::downgrade(&foo);
    /// let other_weak_foo = weak_foo.clone();
    ///
    /// drop(weak_foo);   // Doesn't print anything
    /// drop(foo);        // Prints "dropped!"
    ///
    /// assert!(other_weak_foo.upgrade().is_none());
    /// ```
    fn drop(&mut self) {
        let ptr = *self.ptr;

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
    /// Equality for two `Arc`s.
    ///
    /// Two `Arc`s are equal if their inner values are equal.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::sync::Arc;
    ///
    /// let five = Arc::new(5);
    ///
    /// assert!(five == Arc::new(5));
    /// ```
    fn eq(&self, other: &Arc<T>) -> bool {
        *(*self) == *(*other)
    }

    /// Inequality for two `Arc`s.
    ///
    /// Two `Arc`s are unequal if their inner values are unequal.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::sync::Arc;
    ///
    /// let five = Arc::new(5);
    ///
    /// assert!(five != Arc::new(6));
    /// ```
    fn ne(&self, other: &Arc<T>) -> bool {
        *(*self) != *(*other)
    }
}
#[stable(feature = "rust1", since = "1.0.0")]
impl<T: ?Sized + PartialOrd> PartialOrd for Arc<T> {
    /// Partial comparison for two `Arc`s.
    ///
    /// The two are compared by calling `partial_cmp()` on their inner values.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::sync::Arc;
    /// use std::cmp::Ordering;
    ///
    /// let five = Arc::new(5);
    ///
    /// assert_eq!(Some(Ordering::Less), five.partial_cmp(&Arc::new(6)));
    /// ```
    fn partial_cmp(&self, other: &Arc<T>) -> Option<Ordering> {
        (**self).partial_cmp(&**other)
    }

    /// Less-than comparison for two `Arc`s.
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
    /// assert!(five < Arc::new(6));
    /// ```
    fn lt(&self, other: &Arc<T>) -> bool {
        *(*self) < *(*other)
    }

    /// 'Less than or equal to' comparison for two `Arc`s.
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
    /// assert!(five <= Arc::new(5));
    /// ```
    fn le(&self, other: &Arc<T>) -> bool {
        *(*self) <= *(*other)
    }

    /// Greater-than comparison for two `Arc`s.
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
    /// assert!(five > Arc::new(4));
    /// ```
    fn gt(&self, other: &Arc<T>) -> bool {
        *(*self) > *(*other)
    }

    /// 'Greater than or equal to' comparison for two `Arc`s.
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
    /// assert!(five >= Arc::new(5));
    /// ```
    fn ge(&self, other: &Arc<T>) -> bool {
        *(*self) >= *(*other)
    }
}
#[stable(feature = "rust1", since = "1.0.0")]
impl<T: ?Sized + Ord> Ord for Arc<T> {
    /// Comparison for two `Arc`s.
    ///
    /// The two are compared by calling `cmp()` on their inner values.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::sync::Arc;
    /// use std::cmp::Ordering;
    ///
    /// let five = Arc::new(5);
    ///
    /// assert_eq!(Ordering::Less, five.cmp(&Arc::new(6)));
    /// ```
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
impl<T: ?Sized> fmt::Pointer for Arc<T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        fmt::Pointer::fmt(&*self.ptr, f)
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<T: Default> Default for Arc<T> {
    /// Creates a new `Arc<T>`, with the `Default` value for `T`.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::sync::Arc;
    ///
    /// let x: Arc<i32> = Default::default();
    /// assert_eq!(*x, 0);
    /// ```
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
    use std::option::Option::{None, Some};
    use std::sync::atomic;
    use std::sync::atomic::Ordering::{Acquire, SeqCst};
    use std::thread;
    use std::vec::Vec;
    use super::{Arc, Weak};
    use std::sync::Mutex;
    use std::convert::From;

    struct Canary(*mut atomic::AtomicUsize);

    impl Drop for Canary {
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
    #[cfg_attr(target_os = "emscripten", ignore)]
    fn manually_share_arc() {
        let v = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
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
    fn into_from_raw() {
        let x = Arc::new(box "hello");
        let y = x.clone();

        let x_ptr = Arc::into_raw(x);
        drop(y);
        unsafe {
            assert_eq!(**x_ptr, "hello");

            let x = Arc::from_raw(x_ptr);
            assert_eq!(**x, "hello");

            assert_eq!(Arc::try_unwrap(x).map(|x| *x), Ok("hello"));
        }
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
        let a = Arc::new(0);
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
        let a = Arc::new(0);
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
        let a = Arc::new(5);
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

    #[test]
    fn test_new_weak() {
        let foo: Weak<usize> = Weak::new();
        assert!(foo.upgrade().is_none());
    }

    #[test]
    fn test_ptr_eq() {
        let five = Arc::new(5);
        let same_five = five.clone();
        let other_five = Arc::new(5);

        assert!(Arc::ptr_eq(&five, &same_five));
        assert!(!Arc::ptr_eq(&five, &other_five));
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
