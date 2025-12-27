#![stable(feature = "rust1", since = "1.0.0")]

//! Thread-safe reference-counting pointers.
//!
//! See the [`Arc<T>`][Arc] documentation for more details.
//!
//! **Note**: This module is only available on platforms that support atomic
//! loads and stores of pointers. This may be detected at compile time using
//! `#[cfg(target_has_atomic = "ptr")]`.

use core::alloc::Layout;
use core::any::Any;
use core::cell::CloneFromCell;
#[cfg(not(no_global_oom_handling))]
use core::cell::UnsafeCell;
use core::clone::{CloneToUninit, UseCloned};
use core::cmp::Ordering;
use core::hash::{Hash, Hasher};
use core::marker::Unsize;
#[cfg(not(no_global_oom_handling))]
use core::mem::MaybeUninit;
use core::mem::{self, ManuallyDrop};
use core::ops::{CoerceUnsized, Deref, DerefMut, DerefPure, DispatchFromDyn, LegacyReceiver};
#[cfg(not(no_global_oom_handling))]
use core::ops::{ControlFlow, FromResidual, Residual, Try};
use core::panic::{RefUnwindSafe, UnwindSafe};
use core::pin::{Pin, PinCoerceUnsized};
use core::ptr::{self, NonNull};
use core::sync::atomic::Ordering::{Acquire, Relaxed, Release};
use core::sync::atomic::{self, Atomic};
use core::{borrow, fmt, hint, intrinsics};

use crate::alloc::{AllocError, Allocator, Global};
use crate::borrow::{Cow, ToOwned};
#[cfg(not(no_global_oom_handling))]
use crate::boxed::Box;
#[cfg(not(no_global_oom_handling))]
use crate::raw_rc::MakeMutStrategy;
#[cfg(not(no_global_oom_handling))]
use crate::raw_rc::RefCounts;
use crate::raw_rc::{self, RawRc, RawUniqueRc, RawWeak};
#[cfg(not(no_global_oom_handling))]
use crate::string::String;
#[cfg(not(no_global_oom_handling))]
use crate::vec::Vec;

/// A soft limit on the amount of references that may be made to an `Arc`.
///
/// Going above this limit will abort your program (although not
/// necessarily) at _exactly_ `MAX_REFCOUNT + 1` references.
/// Trying to go above it might call a `panic` (if not actually going above it).
///
/// This is a global invariant, and also applies when using a compare-exchange loop.
///
/// See comment in `Arc::clone`.
const MAX_REFCOUNT: usize = (isize::MAX) as usize;

/// The error in case either counter reaches above `MAX_REFCOUNT`, and we can `panic` safely.
const INTERNAL_OVERFLOW_ERROR: &str = "Arc counter overflow";

#[cfg(not(sanitize = "thread"))]
macro_rules! acquire {
    ($x:expr) => {
        atomic::fence(Acquire)
    };
}

// ThreadSanitizer does not support memory fences. To avoid false positive
// reports in Arc / Weak implementation use atomic loads for synchronization
// instead.
#[cfg(sanitize = "thread")]
macro_rules! acquire {
    ($x:expr) => {
        $x.load(Acquire)
    };
}

type RefCounter = Atomic<usize>;

unsafe impl raw_rc::RefCounter for core::sync::atomic::AtomicUsize {
    #[inline]
    fn increment(&self) {
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
        let old_size = self.fetch_add(1, Relaxed);

        // However we need to guard against massive refcounts in case someone is `mem::forget`ing
        // Arcs. If we don't do this the count can overflow and users will use-after free. This
        // branch will never be taken in any realistic program. We abort because such a program is
        // incredibly degenerate, and we don't care to support it.
        //
        // This check is not 100% water-proof: we error when the refcount grows beyond `isize::MAX`.
        // But we do that check *after* having done the increment, so there is a chance here that
        // the worst already happened and we actually do overflow the `usize` counter. However, that
        // requires the counter to grow from `isize::MAX` to `usize::MAX` between the increment
        // above and the `abort` below, which seems exceedingly unlikely.
        //
        // This is a global invariant, and also applies when using a compare-exchange loop to increment
        // counters in other methods.
        // Otherwise, the counter could be brought to an almost-overflow using a compare-exchange loop,
        // and then overflow using a few `fetch_add`s.
        if old_size > MAX_REFCOUNT {
            intrinsics::abort();
        }
    }

    #[inline]
    fn decrement(&self) -> bool {
        if self.fetch_sub(1, Release) == 1 {
            acquire!(self);

            true
        } else {
            false
        }
    }

    #[inline]
    fn try_upgrade(&self) -> bool {
        #[inline]
        fn checked_increment(n: usize) -> Option<usize> {
            // Any write of 0 we can observe leaves the field in permanently zero state.
            if n == 0 {
                return None;
            }
            // See comments in `RefCounter::increment_ref_count` for why we do this (for `mem::forget`).
            assert!(n <= MAX_REFCOUNT, "{}", INTERNAL_OVERFLOW_ERROR);
            Some(n + 1)
        }

        // We use a CAS loop to increment the strong count instead of a
        // `fetch_add` as this function should never take the reference count
        // from zero to one.
        //
        // `Relaxed` is fine for the failure case because we don't have any expectations about the new state.
        // Acquire is necessary for the success case to synchronise with `Arc::new_cyclic`, when the inner
        // value can be initialized after `Weak` references have already been created. In that case, we
        // expect to observe the fully initialized value.
        self.fetch_update(Acquire, Relaxed, checked_increment).is_ok()
    }

    #[inline]
    fn downgrade_increment_weak(&self) {
        // This Relaxed is OK because we're checking the value in the CAS
        // below.
        let mut cur = self.load(Relaxed);

        loop {
            // check if the weak counter is currently "locked"; if so, spin.
            if cur == usize::MAX {
                hint::spin_loop();
                cur = self.load(Relaxed);

                continue;
            }

            // We can't allow the refcount to increase much past `MAX_REFCOUNT`.
            assert!(cur <= MAX_REFCOUNT, "{}", INTERNAL_OVERFLOW_ERROR);

            // NOTE: this code currently ignores the possibility of overflow
            // into usize::MAX; in general both Rc and Arc need to be adjusted
            // to deal with overflow.

            // Unlike with Clone(), we need this to be an Acquire read to
            // synchronize with the write coming from `is_unique`, so that the
            // events prior to that write happen before this read.
            match self.compare_exchange_weak(cur, cur + 1, Acquire, Relaxed) {
                Ok(_) => break,
                Err(old) => cur = old,
            }
        }
    }

    #[inline]
    fn try_lock_strong_count(&self) -> bool {
        match self.compare_exchange(1, 0, Relaxed, Relaxed) {
            Ok(_) => {
                acquire!(self);

                true
            }
            Err(_) => false,
        }
    }

    #[inline]
    fn unlock_strong_count(&self) {
        self.store(1, Release);
    }

    #[inline]
    fn is_unique(strong_count: &Self, weak_count: &Self) -> bool {
        // lock the weak pointer count if we appear to be the sole weak pointer
        // holder.
        //
        // The acquire label here ensures a happens-before relationship with any
        // writes to `strong` (in particular in `Weak::upgrade`) prior to decrements
        // of the `weak` count (via `Weak::drop`, which uses release). If the upgraded
        // weak ref was never dropped, the CAS here will fail so we do not care to synchronize.
        if weak_count.compare_exchange(1, usize::MAX, Acquire, Relaxed).is_ok() {
            // This needs to be an `Acquire` to synchronize with the decrement of the `strong`
            // counter in `drop` -- the only access that happens when any but the last reference
            // is being dropped.
            let unique = strong_count.load(Acquire) == 1;

            // The release write here synchronizes with a read in `downgrade`,
            // effectively preventing the above read of `strong` from happening
            // after the write.
            weak_count.store(1, Release); // release the lock
            unique
        } else {
            false
        }
    }

    #[cfg(not(no_global_oom_handling))]
    #[inline]
    fn make_mut(strong_count: &Self, weak_count: &Self) -> Option<MakeMutStrategy> {
        // Note that we hold both a strong reference and a weak reference.
        // Thus, releasing our strong reference only will not, by itself, cause
        // the memory to be deallocated.
        //
        // Use Acquire to ensure that we see any writes to `weak` that happen
        // before release writes (i.e., decrements) to `strong`. Since we hold a
        // weak count, there's no chance the allocation itself could be
        // deallocated.
        if strong_count.compare_exchange(1, 0, Acquire, Relaxed).is_ok() {
            if weak_count.load(Relaxed) == 1 {
                // We were the sole reference of either kind; bump back up the
                // strong ref count.
                strong_count.store(1, Release);

                None
            } else {
                // Relaxed suffices in the above because this is fundamentally an
                // optimization: we are always racing with weak pointers being
                // dropped. Worst case, we end up allocated a new Arc unnecessarily.

                // We removed the last strong ref, but there are additional weak
                // refs remaining. We'll move the contents to a new Arc, and
                // invalidate the other weak refs.

                // Note that it is not possible for the read of `weak` to yield
                // usize::MAX (i.e., locked), since the weak count can only be
                // locked by a thread with a strong reference.

                Some(MakeMutStrategy::Move)
            }
        } else {
            Some(MakeMutStrategy::Clone)
        }
    }

    #[cfg(not(no_global_oom_handling))]
    #[inline]
    fn unique_rc_weak_count(weak_count: &Self) -> usize {
        weak_count.load(Acquire)
    }
}

#[cfg(not(no_global_oom_handling))]
#[inline]
fn weak_fn_to_raw_weak_fn<F, T, A>(f: F) -> impl FnOnce(&RawWeak<T, A>) -> T
where
    F: FnOnce(&Weak<T, A>) -> T,
    A: Allocator,
{
    move |raw_weak: &RawWeak<T, A>| f(Weak::ref_from_raw_weak(raw_weak))
}

/// A thread-safe reference-counting pointer. 'Arc' stands for 'Atomically
/// Reference Counted'.
///
/// The type `Arc<T>` provides shared ownership of a value of type `T`,
/// allocated in the heap. Invoking [`clone`][clone] on `Arc` produces
/// a new `Arc` instance, which points to the same allocation on the heap as the
/// source `Arc`, while increasing a reference count. When the last `Arc`
/// pointer to a given allocation is destroyed, the value stored in that allocation (often
/// referred to as "inner value") is also dropped.
///
/// Shared references in Rust disallow mutation by default, and `Arc` is no
/// exception: you cannot generally obtain a mutable reference to something
/// inside an `Arc`. If you do need to mutate through an `Arc`, you have several options:
///
/// 1. Use interior mutability with synchronization primitives like [`Mutex`][mutex],
///    [`RwLock`][rwlock], or one of the [`Atomic`][atomic] types.
///
/// 2. Use clone-on-write semantics with [`Arc::make_mut`] which provides efficient mutation
///    without requiring interior mutability. This approach clones the data only when
///    needed (when there are multiple references) and can be more efficient when mutations
///    are infrequent.
///
/// 3. Use [`Arc::get_mut`] when you know your `Arc` is not shared (has a reference count of 1),
///    which provides direct mutable access to the inner value without any cloning.
///
/// ```
/// use std::sync::Arc;
///
/// let mut data = Arc::new(vec![1, 2, 3]);
///
/// // This will clone the vector only if there are other references to it
/// Arc::make_mut(&mut data).push(4);
///
/// assert_eq!(*data, vec![1, 2, 3, 4]);
/// ```
///
/// **Note**: This type is only available on platforms that support atomic
/// loads and stores of pointers, which includes all platforms that support
/// the `std` crate but not all those which only support [`alloc`](crate).
/// This may be detected at compile time using `#[cfg(target_has_atomic = "ptr")]`.
///
/// ## Thread Safety
///
/// Unlike [`Rc<T>`], `Arc<T>` uses atomic operations for its reference
/// counting. This means that it is thread-safe. The disadvantage is that
/// atomic operations are more expensive than ordinary memory accesses. If you
/// are not sharing reference-counted allocations between threads, consider using
/// [`Rc<T>`] for lower overhead. [`Rc<T>`] is a safe default, because the
/// compiler will catch any attempt to send an [`Rc<T>`] between threads.
/// However, a library might choose `Arc<T>` in order to give library consumers
/// more flexibility.
///
/// `Arc<T>` will implement [`Send`] and [`Sync`] as long as the `T` implements
/// [`Send`] and [`Sync`]. Why can't you put a non-thread-safe type `T` in an
/// `Arc<T>` to make it thread-safe? This may be a bit counter-intuitive at
/// first: after all, isn't the point of `Arc<T>` thread safety? The key is
/// this: `Arc<T>` makes it thread safe to have multiple ownership of the same
/// data, but it  doesn't add thread safety to its data. Consider
/// <code>Arc<[RefCell\<T>]></code>. [`RefCell<T>`] isn't [`Sync`], and if `Arc<T>` was always
/// [`Send`], <code>Arc<[RefCell\<T>]></code> would be as well. But then we'd have a problem:
/// [`RefCell<T>`] is not thread safe; it keeps track of the borrowing count using
/// non-atomic operations.
///
/// In the end, this means that you may need to pair `Arc<T>` with some sort of
/// [`std::sync`] type, usually [`Mutex<T>`][mutex].
///
/// ## Breaking cycles with `Weak`
///
/// The [`downgrade`][downgrade] method can be used to create a non-owning
/// [`Weak`] pointer. A [`Weak`] pointer can be [`upgrade`][upgrade]d
/// to an `Arc`, but this will return [`None`] if the value stored in the allocation has
/// already been dropped. In other words, `Weak` pointers do not keep the value
/// inside the allocation alive; however, they *do* keep the allocation
/// (the backing store for the value) alive.
///
/// A cycle between `Arc` pointers will never be deallocated. For this reason,
/// [`Weak`] is used to break cycles. For example, a tree could have
/// strong `Arc` pointers from parent nodes to children, and [`Weak`]
/// pointers from children back to their parents.
///
/// # Cloning references
///
/// Creating a new reference from an existing reference-counted pointer is done using the
/// `Clone` trait implemented for [`Arc<T>`][Arc] and [`Weak<T>`][Weak].
///
/// ```
/// use std::sync::Arc;
/// let foo = Arc::new(vec![1.0, 2.0, 3.0]);
/// // The two syntaxes below are equivalent.
/// let a = foo.clone();
/// let b = Arc::clone(&foo);
/// // a, b, and foo are all Arcs that point to the same memory location
/// ```
///
/// ## `Deref` behavior
///
/// `Arc<T>` automatically dereferences to `T` (via the [`Deref`] trait),
/// so you can call `T`'s methods on a value of type `Arc<T>`. To avoid name
/// clashes with `T`'s methods, the methods of `Arc<T>` itself are associated
/// functions, called using [fully qualified syntax]:
///
/// ```
/// use std::sync::Arc;
///
/// let my_arc = Arc::new(());
/// let my_weak = Arc::downgrade(&my_arc);
/// ```
///
/// `Arc<T>`'s implementations of traits like `Clone` may also be called using
/// fully qualified syntax. Some people prefer to use fully qualified syntax,
/// while others prefer using method-call syntax.
///
/// ```
/// use std::sync::Arc;
///
/// let arc = Arc::new(());
/// // Method-call syntax
/// let arc2 = arc.clone();
/// // Fully qualified syntax
/// let arc3 = Arc::clone(&arc);
/// ```
///
/// [`Weak<T>`][Weak] does not auto-dereference to `T`, because the inner value may have
/// already been dropped.
///
/// [`Rc<T>`]: crate::rc::Rc
/// [clone]: Clone::clone
/// [mutex]: ../../std/sync/struct.Mutex.html
/// [rwlock]: ../../std/sync/struct.RwLock.html
/// [atomic]: core::sync::atomic
/// [downgrade]: Arc::downgrade
/// [upgrade]: Weak::upgrade
/// [RefCell\<T>]: core::cell::RefCell
/// [`RefCell<T>`]: core::cell::RefCell
/// [`std::sync`]: ../../std/sync/index.html
/// [`Arc::clone(&from)`]: Arc::clone
/// [fully qualified syntax]: https://doc.rust-lang.org/book/ch19-03-advanced-traits.html#fully-qualified-syntax-for-disambiguation-calling-methods-with-the-same-name
///
/// # Examples
///
/// Sharing some immutable data between threads:
///
/// ```
/// use std::sync::Arc;
/// use std::thread;
///
/// let five = Arc::new(5);
///
/// for _ in 0..10 {
///     let five = Arc::clone(&five);
///
///     thread::spawn(move || {
///         println!("{five:?}");
///     });
/// }
/// ```
///
/// Sharing a mutable [`AtomicUsize`]:
///
/// [`AtomicUsize`]: core::sync::atomic::AtomicUsize "sync::atomic::AtomicUsize"
///
/// ```
/// use std::sync::Arc;
/// use std::sync::atomic::{AtomicUsize, Ordering};
/// use std::thread;
///
/// let val = Arc::new(AtomicUsize::new(5));
///
/// for _ in 0..10 {
///     let val = Arc::clone(&val);
///
///     thread::spawn(move || {
///         let v = val.fetch_add(1, Ordering::Relaxed);
///         println!("{v:?}");
///     });
/// }
/// ```
///
/// See the [`rc` documentation][rc_examples] for more examples of reference
/// counting in general.
///
/// [rc_examples]: crate::rc#examples
#[doc(search_unbox)]
#[rustc_diagnostic_item = "Arc"]
#[stable(feature = "rust1", since = "1.0.0")]
#[rustc_insignificant_dtor]
pub struct Arc<
    T: ?Sized,
    #[unstable(feature = "allocator_api", issue = "32838")] A: Allocator = Global,
> {
    raw_rc: RawRc<T, A>,
}

#[stable(feature = "rust1", since = "1.0.0")]
unsafe impl<T: ?Sized + Sync + Send, A: Allocator + Send> Send for Arc<T, A> {}
#[stable(feature = "rust1", since = "1.0.0")]
unsafe impl<T: ?Sized + Sync + Send, A: Allocator + Sync> Sync for Arc<T, A> {}

#[stable(feature = "catch_unwind", since = "1.9.0")]
impl<T: RefUnwindSafe + ?Sized, A: Allocator + UnwindSafe> UnwindSafe for Arc<T, A> {}

#[unstable(feature = "coerce_unsized", issue = "18598")]
impl<T: ?Sized + Unsize<U>, U: ?Sized, A: Allocator> CoerceUnsized<Arc<U, A>> for Arc<T, A> {}

#[unstable(feature = "dispatch_from_dyn", issue = "none")]
impl<T: ?Sized + Unsize<U>, U: ?Sized> DispatchFromDyn<Arc<U>> for Arc<T> {}

// SAFETY: `Arc::clone` doesn't access any `Cell`s which could contain the `Arc` being cloned.
#[unstable(feature = "cell_get_cloned", issue = "145329")]
unsafe impl<T: ?Sized> CloneFromCell for Arc<T> {}

/// `Weak` is a version of [`Arc`] that holds a non-owning reference to the
/// managed allocation.
///
/// The allocation is accessed by calling [`upgrade`] on the `Weak`
/// pointer, which returns an <code>[Option]<[Arc]\<T>></code>.
///
/// Since a `Weak` reference does not count towards ownership, it will not
/// prevent the value stored in the allocation from being dropped, and `Weak` itself makes no
/// guarantees about the value still being present. Thus it may return [`None`]
/// when [`upgrade`]d. Note however that a `Weak` reference *does* prevent the allocation
/// itself (the backing store) from being deallocated.
///
/// A `Weak` pointer is useful for keeping a temporary reference to the allocation
/// managed by [`Arc`] without preventing its inner value from being dropped. It is also used to
/// prevent circular references between [`Arc`] pointers, since mutual owning references
/// would never allow either [`Arc`] to be dropped. For example, a tree could
/// have strong [`Arc`] pointers from parent nodes to children, and `Weak`
/// pointers from children back to their parents.
///
/// The typical way to obtain a `Weak` pointer is to call [`Arc::downgrade`].
///
/// [`upgrade`]: Weak::upgrade
#[stable(feature = "arc_weak", since = "1.4.0")]
#[rustc_diagnostic_item = "ArcWeak"]
pub struct Weak<
    T: ?Sized,
    #[unstable(feature = "allocator_api", issue = "32838")] A: Allocator = Global,
> {
    raw_weak: RawWeak<T, A>,
}

#[stable(feature = "arc_weak", since = "1.4.0")]
unsafe impl<T: ?Sized + Sync + Send, A: Allocator + Send> Send for Weak<T, A> {}
#[stable(feature = "arc_weak", since = "1.4.0")]
unsafe impl<T: ?Sized + Sync + Send, A: Allocator + Sync> Sync for Weak<T, A> {}

#[unstable(feature = "coerce_unsized", issue = "18598")]
impl<T: ?Sized + Unsize<U>, U: ?Sized, A: Allocator> CoerceUnsized<Weak<U, A>> for Weak<T, A> {}
#[unstable(feature = "dispatch_from_dyn", issue = "none")]
impl<T: ?Sized + Unsize<U>, U: ?Sized> DispatchFromDyn<Weak<U>> for Weak<T> {}

// SAFETY: `Weak::clone` doesn't access any `Cell`s which could contain the `Weak` being cloned.
#[unstable(feature = "cell_get_cloned", issue = "145329")]
unsafe impl<T: ?Sized> CloneFromCell for Weak<T> {}

#[stable(feature = "arc_weak", since = "1.4.0")]
impl<T: ?Sized, A: Allocator> fmt::Debug for Weak<T, A> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        <RawWeak<T, A> as fmt::Debug>::fmt(&self.raw_weak, f)
    }
}

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
    #[cfg(not(no_global_oom_handling))]
    #[inline]
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn new(data: T) -> Arc<T> {
        Self { raw_rc: RawRc::new(data) }
    }

    /// Constructs a new `Arc<T>` while giving you a `Weak<T>` to the allocation,
    /// to allow you to construct a `T` which holds a weak pointer to itself.
    ///
    /// Generally, a structure circularly referencing itself, either directly or
    /// indirectly, should not hold a strong reference to itself to prevent a memory leak.
    /// Using this function, you get access to the weak pointer during the
    /// initialization of `T`, before the `Arc<T>` is created, such that you can
    /// clone and store it inside the `T`.
    ///
    /// `new_cyclic` first allocates the managed allocation for the `Arc<T>`,
    /// then calls your closure, giving it a `Weak<T>` to this allocation,
    /// and only afterwards completes the construction of the `Arc<T>` by placing
    /// the `T` returned from your closure into the allocation.
    ///
    /// Since the new `Arc<T>` is not fully-constructed until `Arc<T>::new_cyclic`
    /// returns, calling [`upgrade`] on the weak reference inside your closure will
    /// fail and result in a `None` value.
    ///
    /// # Panics
    ///
    /// If `data_fn` panics, the panic is propagated to the caller, and the
    /// temporary [`Weak<T>`] is dropped normally.
    ///
    /// # Example
    ///
    /// ```
    /// # #![allow(dead_code)]
    /// use std::sync::{Arc, Weak};
    ///
    /// struct Gadget {
    ///     me: Weak<Gadget>,
    /// }
    ///
    /// impl Gadget {
    ///     /// Constructs a reference counted Gadget.
    ///     fn new() -> Arc<Self> {
    ///         // `me` is a `Weak<Gadget>` pointing at the new allocation of the
    ///         // `Arc` we're constructing.
    ///         Arc::new_cyclic(|me| {
    ///             // Create the actual struct here.
    ///             Gadget { me: me.clone() }
    ///         })
    ///     }
    ///
    ///     /// Returns a reference counted pointer to Self.
    ///     fn me(&self) -> Arc<Self> {
    ///         self.me.upgrade().unwrap()
    ///     }
    /// }
    /// ```
    /// [`upgrade`]: Weak::upgrade
    #[cfg(not(no_global_oom_handling))]
    #[inline]
    #[stable(feature = "arc_new_cyclic", since = "1.60.0")]
    pub fn new_cyclic<F>(data_fn: F) -> Arc<T>
    where
        F: FnOnce(&Weak<T>) -> T,
    {
        let data_fn = weak_fn_to_raw_weak_fn(data_fn);
        let raw_rc = unsafe { RawRc::new_cyclic::<_, RefCounter>(data_fn) };

        Self { raw_rc }
    }

    /// Constructs a new `Arc` with uninitialized contents.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::sync::Arc;
    ///
    /// let mut five = Arc::<u32>::new_uninit();
    ///
    /// // Deferred initialization:
    /// Arc::get_mut(&mut five).unwrap().write(5);
    ///
    /// let five = unsafe { five.assume_init() };
    ///
    /// assert_eq!(*five, 5)
    /// ```
    #[cfg(not(no_global_oom_handling))]
    #[inline]
    #[stable(feature = "new_uninit", since = "1.82.0")]
    #[must_use]
    pub fn new_uninit() -> Arc<mem::MaybeUninit<T>> {
        Arc { raw_rc: RawRc::new_uninit() }
    }

    /// Constructs a new `Arc` with uninitialized contents, with the memory
    /// being filled with `0` bytes.
    ///
    /// See [`MaybeUninit::zeroed`][zeroed] for examples of correct and incorrect usage
    /// of this method.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::sync::Arc;
    ///
    /// let zero = Arc::<u32>::new_zeroed();
    /// let zero = unsafe { zero.assume_init() };
    ///
    /// assert_eq!(*zero, 0)
    /// ```
    ///
    /// [zeroed]: mem::MaybeUninit::zeroed
    #[cfg(not(no_global_oom_handling))]
    #[inline]
    #[stable(feature = "new_zeroed_alloc", since = "1.92.0")]
    #[must_use]
    pub fn new_zeroed() -> Arc<mem::MaybeUninit<T>> {
        Arc { raw_rc: RawRc::new_zeroed() }
    }

    /// Constructs a new `Pin<Arc<T>>`. If `T` does not implement `Unpin`, then
    /// `data` will be pinned in memory and unable to be moved.
    #[cfg(not(no_global_oom_handling))]
    #[stable(feature = "pin", since = "1.33.0")]
    #[must_use]
    pub fn pin(data: T) -> Pin<Arc<T>> {
        unsafe { Pin::new_unchecked(Arc::new(data)) }
    }

    /// Constructs a new `Pin<Arc<T>>`, return an error if allocation fails.
    #[unstable(feature = "allocator_api", issue = "32838")]
    #[inline]
    pub fn try_pin(data: T) -> Result<Pin<Arc<T>>, AllocError> {
        unsafe { Ok(Pin::new_unchecked(Arc::try_new(data)?)) }
    }

    /// Constructs a new `Arc<T>`, returning an error if allocation fails.
    ///
    /// # Examples
    ///
    /// ```
    /// #![feature(allocator_api)]
    /// use std::sync::Arc;
    ///
    /// let five = Arc::try_new(5)?;
    /// # Ok::<(), std::alloc::AllocError>(())
    /// ```
    #[unstable(feature = "allocator_api", issue = "32838")]
    #[inline]
    pub fn try_new(data: T) -> Result<Arc<T>, AllocError> {
        RawRc::try_new(data).map(|raw_rc| Self { raw_rc })
    }

    /// Constructs a new `Arc` with uninitialized contents, returning an error
    /// if allocation fails.
    ///
    /// # Examples
    ///
    /// ```
    /// #![feature(allocator_api)]
    ///
    /// use std::sync::Arc;
    ///
    /// let mut five = Arc::<u32>::try_new_uninit()?;
    ///
    /// // Deferred initialization:
    /// Arc::get_mut(&mut five).unwrap().write(5);
    ///
    /// let five = unsafe { five.assume_init() };
    ///
    /// assert_eq!(*five, 5);
    /// # Ok::<(), std::alloc::AllocError>(())
    /// ```
    #[unstable(feature = "allocator_api", issue = "32838")]
    pub fn try_new_uninit() -> Result<Arc<mem::MaybeUninit<T>>, AllocError> {
        RawRc::try_new_uninit().map(|raw_rc| Arc { raw_rc })
    }

    /// Constructs a new `Arc` with uninitialized contents, with the memory
    /// being filled with `0` bytes, returning an error if allocation fails.
    ///
    /// See [`MaybeUninit::zeroed`][zeroed] for examples of correct and incorrect usage
    /// of this method.
    ///
    /// # Examples
    ///
    /// ```
    /// #![feature( allocator_api)]
    ///
    /// use std::sync::Arc;
    ///
    /// let zero = Arc::<u32>::try_new_zeroed()?;
    /// let zero = unsafe { zero.assume_init() };
    ///
    /// assert_eq!(*zero, 0);
    /// # Ok::<(), std::alloc::AllocError>(())
    /// ```
    ///
    /// [zeroed]: mem::MaybeUninit::zeroed
    #[unstable(feature = "allocator_api", issue = "32838")]
    pub fn try_new_zeroed() -> Result<Arc<mem::MaybeUninit<T>>, AllocError> {
        RawRc::try_new_zeroed().map(|raw_rc| Arc { raw_rc })
    }

    /// Maps the value in an `Arc`, reusing the allocation if possible.
    ///
    /// `f` is called on a reference to the value in the `Arc`, and the result is returned, also in
    /// an `Arc`.
    ///
    /// Note: this is an associated function, which means that you have
    /// to call it as `Arc::map(a, f)` instead of `r.map(a)`. This
    /// is so that there is no conflict with a method on the inner type.
    ///
    /// # Examples
    ///
    /// ```
    /// #![feature(smart_pointer_try_map)]
    ///
    /// use std::sync::Arc;
    ///
    /// let r = Arc::new(7);
    /// let new = Arc::map(r, |i| i + 7);
    /// assert_eq!(*new, 14);
    /// ```
    #[cfg(not(no_global_oom_handling))]
    #[unstable(feature = "smart_pointer_try_map", issue = "144419")]
    pub fn map<U>(this: Self, f: impl FnOnce(&T) -> U) -> Arc<U> {
        let raw_rc = Self::into_raw_rc(this);

        Arc { raw_rc: unsafe { raw_rc.map::<RefCounter, U>(f) } }
    }

    /// Attempts to map the value in an `Arc`, reusing the allocation if possible.
    ///
    /// `f` is called on a reference to the value in the `Arc`, and if the operation succeeds, the
    /// result is returned, also in an `Arc`.
    ///
    /// Note: this is an associated function, which means that you have
    /// to call it as `Arc::try_map(a, f)` instead of `a.try_map(f)`. This
    /// is so that there is no conflict with a method on the inner type.
    ///
    /// # Examples
    ///
    /// ```
    /// #![feature(smart_pointer_try_map)]
    ///
    /// use std::sync::Arc;
    ///
    /// let b = Arc::new(7);
    /// let new = Arc::try_map(b, |&i| u32::try_from(i)).unwrap();
    /// assert_eq!(*new, 7);
    /// ```
    #[cfg(not(no_global_oom_handling))]
    #[unstable(feature = "smart_pointer_try_map", issue = "144419")]
    pub fn try_map<R>(
        this: Self,
        f: impl FnOnce(&T) -> R,
    ) -> <R::Residual as Residual<Arc<R::Output>>>::TryType
    where
        R: Try,
        R::Residual: Residual<Arc<R::Output>>,
    {
        let raw_rc = Self::into_raw_rc(this);

        match unsafe { raw_rc.try_map::<RefCounter, R>(f) } {
            ControlFlow::Continue(raw_rc) => Try::from_output(Arc { raw_rc }),
            ControlFlow::Break(residual) => FromResidual::from_residual(residual),
        }
    }
}

impl<T, A: Allocator> Arc<T, A> {
    /// Constructs a new `Arc<T>` in the provided allocator.
    ///
    /// # Examples
    ///
    /// ```
    /// #![feature(allocator_api)]
    ///
    /// use std::sync::Arc;
    /// use std::alloc::System;
    ///
    /// let five = Arc::new_in(5, System);
    /// ```
    #[inline]
    #[cfg(not(no_global_oom_handling))]
    #[unstable(feature = "allocator_api", issue = "32838")]
    pub fn new_in(data: T, alloc: A) -> Arc<T, A> {
        Self { raw_rc: RawRc::new_in(data, alloc) }
    }

    /// Constructs a new `Arc` with uninitialized contents in the provided allocator.
    ///
    /// # Examples
    ///
    /// ```
    /// #![feature(get_mut_unchecked)]
    /// #![feature(allocator_api)]
    ///
    /// use std::sync::Arc;
    /// use std::alloc::System;
    ///
    /// let mut five = Arc::<u32, _>::new_uninit_in(System);
    ///
    /// let five = unsafe {
    ///     // Deferred initialization:
    ///     Arc::get_mut_unchecked(&mut five).as_mut_ptr().write(5);
    ///
    ///     five.assume_init()
    /// };
    ///
    /// assert_eq!(*five, 5)
    /// ```
    #[cfg(not(no_global_oom_handling))]
    #[unstable(feature = "allocator_api", issue = "32838")]
    #[inline]
    pub fn new_uninit_in(alloc: A) -> Arc<mem::MaybeUninit<T>, A> {
        Arc { raw_rc: RawRc::new_uninit_in(alloc) }
    }

    /// Constructs a new `Arc` with uninitialized contents, with the memory
    /// being filled with `0` bytes, in the provided allocator.
    ///
    /// See [`MaybeUninit::zeroed`][zeroed] for examples of correct and incorrect usage
    /// of this method.
    ///
    /// # Examples
    ///
    /// ```
    /// #![feature(allocator_api)]
    ///
    /// use std::sync::Arc;
    /// use std::alloc::System;
    ///
    /// let zero = Arc::<u32, _>::new_zeroed_in(System);
    /// let zero = unsafe { zero.assume_init() };
    ///
    /// assert_eq!(*zero, 0)
    /// ```
    ///
    /// [zeroed]: mem::MaybeUninit::zeroed
    #[cfg(not(no_global_oom_handling))]
    #[unstable(feature = "allocator_api", issue = "32838")]
    #[inline]
    pub fn new_zeroed_in(alloc: A) -> Arc<mem::MaybeUninit<T>, A> {
        Arc { raw_rc: RawRc::new_zeroed_in(alloc) }
    }

    /// Constructs a new `Arc<T, A>` in the given allocator while giving you a `Weak<T, A>` to the allocation,
    /// to allow you to construct a `T` which holds a weak pointer to itself.
    ///
    /// Generally, a structure circularly referencing itself, either directly or
    /// indirectly, should not hold a strong reference to itself to prevent a memory leak.
    /// Using this function, you get access to the weak pointer during the
    /// initialization of `T`, before the `Arc<T, A>` is created, such that you can
    /// clone and store it inside the `T`.
    ///
    /// `new_cyclic_in` first allocates the managed allocation for the `Arc<T, A>`,
    /// then calls your closure, giving it a `Weak<T, A>` to this allocation,
    /// and only afterwards completes the construction of the `Arc<T, A>` by placing
    /// the `T` returned from your closure into the allocation.
    ///
    /// Since the new `Arc<T, A>` is not fully-constructed until `Arc<T, A>::new_cyclic_in`
    /// returns, calling [`upgrade`] on the weak reference inside your closure will
    /// fail and result in a `None` value.
    ///
    /// # Panics
    ///
    /// If `data_fn` panics, the panic is propagated to the caller, and the
    /// temporary [`Weak<T>`] is dropped normally.
    ///
    /// # Example
    ///
    /// See [`new_cyclic`]
    ///
    /// [`new_cyclic`]: Arc::new_cyclic
    /// [`upgrade`]: Weak::upgrade
    #[cfg(not(no_global_oom_handling))]
    #[inline]
    #[unstable(feature = "allocator_api", issue = "32838")]
    pub fn new_cyclic_in<F>(data_fn: F, alloc: A) -> Arc<T, A>
    where
        F: FnOnce(&Weak<T, A>) -> T,
    {
        let data_fn = weak_fn_to_raw_weak_fn(data_fn);
        let raw_rc = unsafe { RawRc::new_cyclic_in::<_, RefCounter>(data_fn, alloc) };

        Self { raw_rc }
    }

    /// Constructs a new `Pin<Arc<T, A>>` in the provided allocator. If `T` does not implement `Unpin`,
    /// then `data` will be pinned in memory and unable to be moved.
    #[cfg(not(no_global_oom_handling))]
    #[unstable(feature = "allocator_api", issue = "32838")]
    #[inline]
    pub fn pin_in(data: T, alloc: A) -> Pin<Arc<T, A>>
    where
        A: 'static,
    {
        unsafe { Pin::new_unchecked(Arc::new_in(data, alloc)) }
    }

    /// Constructs a new `Pin<Arc<T, A>>` in the provided allocator, return an error if allocation
    /// fails.
    #[inline]
    #[unstable(feature = "allocator_api", issue = "32838")]
    pub fn try_pin_in(data: T, alloc: A) -> Result<Pin<Arc<T, A>>, AllocError>
    where
        A: 'static,
    {
        unsafe { Ok(Pin::new_unchecked(Arc::try_new_in(data, alloc)?)) }
    }

    /// Constructs a new `Arc<T, A>` in the provided allocator, returning an error if allocation fails.
    ///
    /// # Examples
    ///
    /// ```
    /// #![feature(allocator_api)]
    ///
    /// use std::sync::Arc;
    /// use std::alloc::System;
    ///
    /// let five = Arc::try_new_in(5, System)?;
    /// # Ok::<(), std::alloc::AllocError>(())
    /// ```
    #[unstable(feature = "allocator_api", issue = "32838")]
    #[inline]
    pub fn try_new_in(data: T, alloc: A) -> Result<Arc<T, A>, AllocError> {
        RawRc::try_new_in(data, alloc).map(|raw_rc| Self { raw_rc })
    }

    /// Constructs a new `Arc` with uninitialized contents, in the provided allocator, returning an
    /// error if allocation fails.
    ///
    /// # Examples
    ///
    /// ```
    /// #![feature(allocator_api)]
    /// #![feature(get_mut_unchecked)]
    ///
    /// use std::sync::Arc;
    /// use std::alloc::System;
    ///
    /// let mut five = Arc::<u32, _>::try_new_uninit_in(System)?;
    ///
    /// let five = unsafe {
    ///     // Deferred initialization:
    ///     Arc::get_mut_unchecked(&mut five).as_mut_ptr().write(5);
    ///
    ///     five.assume_init()
    /// };
    ///
    /// assert_eq!(*five, 5);
    /// # Ok::<(), std::alloc::AllocError>(())
    /// ```
    #[unstable(feature = "allocator_api", issue = "32838")]
    #[inline]
    pub fn try_new_uninit_in(alloc: A) -> Result<Arc<mem::MaybeUninit<T>, A>, AllocError> {
        RawRc::try_new_uninit_in(alloc).map(|raw_rc| Arc { raw_rc })
    }

    /// Constructs a new `Arc` with uninitialized contents, with the memory
    /// being filled with `0` bytes, in the provided allocator, returning an error if allocation
    /// fails.
    ///
    /// See [`MaybeUninit::zeroed`][zeroed] for examples of correct and incorrect usage
    /// of this method.
    ///
    /// # Examples
    ///
    /// ```
    /// #![feature(allocator_api)]
    ///
    /// use std::sync::Arc;
    /// use std::alloc::System;
    ///
    /// let zero = Arc::<u32, _>::try_new_zeroed_in(System)?;
    /// let zero = unsafe { zero.assume_init() };
    ///
    /// assert_eq!(*zero, 0);
    /// # Ok::<(), std::alloc::AllocError>(())
    /// ```
    ///
    /// [zeroed]: mem::MaybeUninit::zeroed
    #[unstable(feature = "allocator_api", issue = "32838")]
    #[inline]
    pub fn try_new_zeroed_in(alloc: A) -> Result<Arc<mem::MaybeUninit<T>, A>, AllocError> {
        RawRc::try_new_zeroed_in(alloc).map(|raw_rc| Arc { raw_rc })
    }
    /// Returns the inner value, if the `Arc` has exactly one strong reference.
    ///
    /// Otherwise, an [`Err`] is returned with the same `Arc` that was
    /// passed in.
    ///
    /// This will succeed even if there are outstanding weak references.
    ///
    /// It is strongly recommended to use [`Arc::into_inner`] instead if you don't
    /// keep the `Arc` in the [`Err`] case.
    /// Immediately dropping the [`Err`]-value, as the expression
    /// `Arc::try_unwrap(this).ok()` does, can cause the strong count to
    /// drop to zero and the inner value of the `Arc` to be dropped.
    /// For instance, if two threads execute such an expression in parallel,
    /// there is a race condition without the possibility of unsafety:
    /// The threads could first both check whether they own the last instance
    /// in `Arc::try_unwrap`, determine that they both do not, and then both
    /// discard and drop their instance in the call to [`ok`][`Result::ok`].
    /// In this scenario, the value inside the `Arc` is safely destroyed
    /// by exactly one of the threads, but neither thread will ever be able
    /// to use the value.
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
    /// let _y = Arc::clone(&x);
    /// assert_eq!(*Arc::try_unwrap(x).unwrap_err(), 4);
    /// ```
    #[inline]
    #[stable(feature = "arc_unique", since = "1.4.0")]
    pub fn try_unwrap(this: Self) -> Result<T, Self> {
        let raw_rc = Self::into_raw_rc(this);
        let result = unsafe { raw_rc.try_unwrap::<RefCounter>() };

        result.map_err(|raw_rc| Self { raw_rc })
    }

    /// Returns the inner value, if the `Arc` has exactly one strong reference.
    ///
    /// Otherwise, [`None`] is returned and the `Arc` is dropped.
    ///
    /// This will succeed even if there are outstanding weak references.
    ///
    /// If `Arc::into_inner` is called on every clone of this `Arc`,
    /// it is guaranteed that exactly one of the calls returns the inner value.
    /// This means in particular that the inner value is not dropped.
    ///
    /// [`Arc::try_unwrap`] is conceptually similar to `Arc::into_inner`, but it
    /// is meant for different use-cases. If used as a direct replacement
    /// for `Arc::into_inner` anyway, such as with the expression
    /// <code>[Arc::try_unwrap]\(this).[ok][Result::ok]()</code>, then it does
    /// **not** give the same guarantee as described in the previous paragraph.
    /// For more information, see the examples below and read the documentation
    /// of [`Arc::try_unwrap`].
    ///
    /// # Examples
    ///
    /// Minimal example demonstrating the guarantee that `Arc::into_inner` gives.
    /// ```
    /// use std::sync::Arc;
    ///
    /// let x = Arc::new(3);
    /// let y = Arc::clone(&x);
    ///
    /// // Two threads calling `Arc::into_inner` on both clones of an `Arc`:
    /// let x_thread = std::thread::spawn(|| Arc::into_inner(x));
    /// let y_thread = std::thread::spawn(|| Arc::into_inner(y));
    ///
    /// let x_inner_value = x_thread.join().unwrap();
    /// let y_inner_value = y_thread.join().unwrap();
    ///
    /// // One of the threads is guaranteed to receive the inner value:
    /// assert!(matches!(
    ///     (x_inner_value, y_inner_value),
    ///     (None, Some(3)) | (Some(3), None)
    /// ));
    /// // The result could also be `(None, None)` if the threads called
    /// // `Arc::try_unwrap(x).ok()` and `Arc::try_unwrap(y).ok()` instead.
    /// ```
    ///
    /// A more practical example demonstrating the need for `Arc::into_inner`:
    /// ```
    /// use std::sync::Arc;
    ///
    /// // Definition of a simple singly linked list using `Arc`:
    /// #[derive(Clone)]
    /// struct LinkedList<T>(Option<Arc<Node<T>>>);
    /// struct Node<T>(T, Option<Arc<Node<T>>>);
    ///
    /// // Dropping a long `LinkedList<T>` relying on the destructor of `Arc`
    /// // can cause a stack overflow. To prevent this, we can provide a
    /// // manual `Drop` implementation that does the destruction in a loop:
    /// impl<T> Drop for LinkedList<T> {
    ///     fn drop(&mut self) {
    ///         let mut link = self.0.take();
    ///         while let Some(arc_node) = link.take() {
    ///             if let Some(Node(_value, next)) = Arc::into_inner(arc_node) {
    ///                 link = next;
    ///             }
    ///         }
    ///     }
    /// }
    ///
    /// // Implementation of `new` and `push` omitted
    /// impl<T> LinkedList<T> {
    ///     /* ... */
    /// #   fn new() -> Self {
    /// #       LinkedList(None)
    /// #   }
    /// #   fn push(&mut self, x: T) {
    /// #       self.0 = Some(Arc::new(Node(x, self.0.take())));
    /// #   }
    /// }
    ///
    /// // The following code could have still caused a stack overflow
    /// // despite the manual `Drop` impl if that `Drop` impl had used
    /// // `Arc::try_unwrap(arc).ok()` instead of `Arc::into_inner(arc)`.
    ///
    /// // Create a long list and clone it
    /// let mut x = LinkedList::new();
    /// let size = 100000;
    /// # let size = if cfg!(miri) { 100 } else { size };
    /// for i in 0..size {
    ///     x.push(i); // Adds i to the front of x
    /// }
    /// let y = x.clone();
    ///
    /// // Drop the clones in parallel
    /// let x_thread = std::thread::spawn(|| drop(x));
    /// let y_thread = std::thread::spawn(|| drop(y));
    /// x_thread.join().unwrap();
    /// y_thread.join().unwrap();
    /// ```
    #[inline]
    #[stable(feature = "arc_into_inner", since = "1.70.0")]
    pub fn into_inner(this: Self) -> Option<T> {
        let raw_rc = Self::into_raw_rc(this);

        unsafe { raw_rc.into_inner::<RefCounter>() }
    }
}

impl<T> Arc<[T]> {
    /// Constructs a new atomically reference-counted slice with uninitialized contents.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::sync::Arc;
    ///
    /// let mut values = Arc::<[u32]>::new_uninit_slice(3);
    ///
    /// // Deferred initialization:
    /// let data = Arc::get_mut(&mut values).unwrap();
    /// data[0].write(1);
    /// data[1].write(2);
    /// data[2].write(3);
    ///
    /// let values = unsafe { values.assume_init() };
    ///
    /// assert_eq!(*values, [1, 2, 3])
    /// ```
    #[cfg(not(no_global_oom_handling))]
    #[inline]
    #[stable(feature = "new_uninit", since = "1.82.0")]
    #[must_use]
    pub fn new_uninit_slice(len: usize) -> Arc<[mem::MaybeUninit<T>]> {
        Arc { raw_rc: RawRc::new_uninit_slice(len) }
    }

    /// Constructs a new atomically reference-counted slice with uninitialized contents, with the memory being
    /// filled with `0` bytes.
    ///
    /// See [`MaybeUninit::zeroed`][zeroed] for examples of correct and
    /// incorrect usage of this method.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::sync::Arc;
    ///
    /// let values = Arc::<[u32]>::new_zeroed_slice(3);
    /// let values = unsafe { values.assume_init() };
    ///
    /// assert_eq!(*values, [0, 0, 0])
    /// ```
    ///
    /// [zeroed]: mem::MaybeUninit::zeroed
    #[cfg(not(no_global_oom_handling))]
    #[inline]
    #[stable(feature = "new_zeroed_alloc", since = "1.92.0")]
    #[must_use]
    pub fn new_zeroed_slice(len: usize) -> Arc<[mem::MaybeUninit<T>]> {
        Arc { raw_rc: RawRc::new_zeroed_slice(len) }
    }

    /// Converts the reference-counted slice into a reference-counted array.
    ///
    /// This operation does not reallocate; the underlying array of the slice is simply reinterpreted as an array type.
    ///
    /// If `N` is not exactly equal to the length of `self`, then this method returns `None`.
    #[unstable(feature = "alloc_slice_into_array", issue = "148082")]
    #[inline]
    #[must_use]
    pub fn into_array<const N: usize>(self) -> Option<Arc<[T; N]>> {
        let raw_rc = Self::into_raw_rc(self);
        let result = unsafe { raw_rc.into_array::<N, RefCounter>() };

        result.map(|raw_rc| Arc { raw_rc })
    }
}

impl<T, A: Allocator> Arc<[T], A> {
    /// Constructs a new atomically reference-counted slice with uninitialized contents in the
    /// provided allocator.
    ///
    /// # Examples
    ///
    /// ```
    /// #![feature(get_mut_unchecked)]
    /// #![feature(allocator_api)]
    ///
    /// use std::sync::Arc;
    /// use std::alloc::System;
    ///
    /// let mut values = Arc::<[u32], _>::new_uninit_slice_in(3, System);
    ///
    /// let values = unsafe {
    ///     // Deferred initialization:
    ///     Arc::get_mut_unchecked(&mut values)[0].as_mut_ptr().write(1);
    ///     Arc::get_mut_unchecked(&mut values)[1].as_mut_ptr().write(2);
    ///     Arc::get_mut_unchecked(&mut values)[2].as_mut_ptr().write(3);
    ///
    ///     values.assume_init()
    /// };
    ///
    /// assert_eq!(*values, [1, 2, 3])
    /// ```
    #[cfg(not(no_global_oom_handling))]
    #[unstable(feature = "allocator_api", issue = "32838")]
    #[inline]
    pub fn new_uninit_slice_in(len: usize, alloc: A) -> Arc<[mem::MaybeUninit<T>], A> {
        Arc { raw_rc: RawRc::new_uninit_slice_in(len, alloc) }
    }

    /// Constructs a new atomically reference-counted slice with uninitialized contents, with the memory being
    /// filled with `0` bytes, in the provided allocator.
    ///
    /// See [`MaybeUninit::zeroed`][zeroed] for examples of correct and
    /// incorrect usage of this method.
    ///
    /// # Examples
    ///
    /// ```
    /// #![feature(allocator_api)]
    ///
    /// use std::sync::Arc;
    /// use std::alloc::System;
    ///
    /// let values = Arc::<[u32], _>::new_zeroed_slice_in(3, System);
    /// let values = unsafe { values.assume_init() };
    ///
    /// assert_eq!(*values, [0, 0, 0])
    /// ```
    ///
    /// [zeroed]: mem::MaybeUninit::zeroed
    #[cfg(not(no_global_oom_handling))]
    #[unstable(feature = "allocator_api", issue = "32838")]
    #[inline]
    pub fn new_zeroed_slice_in(len: usize, alloc: A) -> Arc<[mem::MaybeUninit<T>], A> {
        Arc { raw_rc: RawRc::new_zeroed_slice_in(len, alloc) }
    }
}

impl<T, A: Allocator> Arc<mem::MaybeUninit<T>, A> {
    /// Converts to `Arc<T>`.
    ///
    /// # Safety
    ///
    /// As with [`MaybeUninit::assume_init`],
    /// it is up to the caller to guarantee that the inner value
    /// really is in an initialized state.
    /// Calling this when the content is not yet fully initialized
    /// causes immediate undefined behavior.
    ///
    /// [`MaybeUninit::assume_init`]: mem::MaybeUninit::assume_init
    ///
    /// # Examples
    ///
    /// ```
    /// use std::sync::Arc;
    ///
    /// let mut five = Arc::<u32>::new_uninit();
    ///
    /// // Deferred initialization:
    /// Arc::get_mut(&mut five).unwrap().write(5);
    ///
    /// let five = unsafe { five.assume_init() };
    ///
    /// assert_eq!(*five, 5)
    /// ```
    #[stable(feature = "new_uninit", since = "1.82.0")]
    #[must_use = "`self` will be dropped if the result is not used"]
    #[inline]
    pub unsafe fn assume_init(self) -> Arc<T, A> {
        let raw_rc = Self::into_raw_rc(self);
        let raw_rc = unsafe { raw_rc.assume_init() };

        Arc { raw_rc }
    }
}

impl<T: ?Sized + CloneToUninit> Arc<T> {
    /// Constructs a new `Arc<T>` with a clone of `value`.
    ///
    /// # Examples
    ///
    /// ```
    /// #![feature(clone_from_ref)]
    /// use std::sync::Arc;
    ///
    /// let hello: Arc<str> = Arc::clone_from_ref("hello");
    /// ```
    #[cfg(not(no_global_oom_handling))]
    #[unstable(feature = "clone_from_ref", issue = "149075")]
    pub fn clone_from_ref(value: &T) -> Arc<T> {
        Self { raw_rc: RawRc::clone_from_ref(value) }
    }

    /// Constructs a new `Arc<T>` with a clone of `value`, returning an error if allocation fails
    ///
    /// # Examples
    ///
    /// ```
    /// #![feature(clone_from_ref)]
    /// #![feature(allocator_api)]
    /// use std::sync::Arc;
    ///
    /// let hello: Arc<str> = Arc::try_clone_from_ref("hello")?;
    /// # Ok::<(), std::alloc::AllocError>(())
    /// ```
    #[unstable(feature = "clone_from_ref", issue = "149075")]
    //#[unstable(feature = "allocator_api", issue = "32838")]
    pub fn try_clone_from_ref(value: &T) -> Result<Arc<T>, AllocError> {
        RawRc::try_clone_from_ref(value).map(|raw_rc| Self { raw_rc })
    }
}

impl<T: ?Sized + CloneToUninit, A: Allocator> Arc<T, A> {
    /// Constructs a new `Arc<T>` with a clone of `value` in the provided allocator.
    ///
    /// # Examples
    ///
    /// ```
    /// #![feature(clone_from_ref)]
    /// #![feature(allocator_api)]
    /// use std::sync::Arc;
    /// use std::alloc::System;
    ///
    /// let hello: Arc<str, System> = Arc::clone_from_ref_in("hello", System);
    /// ```
    #[cfg(not(no_global_oom_handling))]
    #[unstable(feature = "clone_from_ref", issue = "149075")]
    //#[unstable(feature = "allocator_api", issue = "32838")]
    pub fn clone_from_ref_in(value: &T, alloc: A) -> Arc<T, A> {
        Self { raw_rc: RawRc::clone_from_ref_in(value, alloc) }
    }

    /// Constructs a new `Arc<T>` with a clone of `value` in the provided allocator, returning an error if allocation fails
    ///
    /// # Examples
    ///
    /// ```
    /// #![feature(clone_from_ref)]
    /// #![feature(allocator_api)]
    /// use std::sync::Arc;
    /// use std::alloc::System;
    ///
    /// let hello: Arc<str, System> = Arc::try_clone_from_ref_in("hello", System)?;
    /// # Ok::<(), std::alloc::AllocError>(())
    /// ```
    #[unstable(feature = "clone_from_ref", issue = "149075")]
    //#[unstable(feature = "allocator_api", issue = "32838")]
    pub fn try_clone_from_ref_in(value: &T, alloc: A) -> Result<Arc<T, A>, AllocError> {
        RawRc::try_clone_from_ref_in(value, alloc).map(|raw_rc| Self { raw_rc })
    }
}

impl<T, A: Allocator> Arc<[mem::MaybeUninit<T>], A> {
    /// Converts to `Arc<[T]>`.
    ///
    /// # Safety
    ///
    /// As with [`MaybeUninit::assume_init`],
    /// it is up to the caller to guarantee that the inner value
    /// really is in an initialized state.
    /// Calling this when the content is not yet fully initialized
    /// causes immediate undefined behavior.
    ///
    /// [`MaybeUninit::assume_init`]: mem::MaybeUninit::assume_init
    ///
    /// # Examples
    ///
    /// ```
    /// use std::sync::Arc;
    ///
    /// let mut values = Arc::<[u32]>::new_uninit_slice(3);
    ///
    /// // Deferred initialization:
    /// let data = Arc::get_mut(&mut values).unwrap();
    /// data[0].write(1);
    /// data[1].write(2);
    /// data[2].write(3);
    ///
    /// let values = unsafe { values.assume_init() };
    ///
    /// assert_eq!(*values, [1, 2, 3])
    /// ```
    #[stable(feature = "new_uninit", since = "1.82.0")]
    #[must_use = "`self` will be dropped if the result is not used"]
    #[inline]
    pub unsafe fn assume_init(self) -> Arc<[T], A> {
        let raw_rc = Self::into_raw_rc(self);
        let raw_rc = unsafe { raw_rc.assume_init() };

        Arc { raw_rc }
    }
}

impl<T: ?Sized> Arc<T> {
    /// Constructs an `Arc<T>` from a raw pointer.
    ///
    /// The raw pointer must have been previously returned by a call to
    /// [`Arc<U>::into_raw`][into_raw] with the following requirements:
    ///
    /// * If `U` is sized, it must have the same size and alignment as `T`. This
    ///   is trivially true if `U` is `T`.
    /// * If `U` is unsized, its data pointer must have the same size and
    ///   alignment as `T`. This is trivially true if `Arc<U>` was constructed
    ///   through `Arc<T>` and then converted to `Arc<U>` through an [unsized
    ///   coercion].
    ///
    /// Note that if `U` or `U`'s data pointer is not `T` but has the same size
    /// and alignment, this is basically like transmuting references of
    /// different types. See [`mem::transmute`][transmute] for more information
    /// on what restrictions apply in this case.
    ///
    /// The raw pointer must point to a block of memory allocated by the global allocator.
    ///
    /// The user of `from_raw` has to make sure a specific value of `T` is only
    /// dropped once.
    ///
    /// This function is unsafe because improper use may lead to memory unsafety,
    /// even if the returned `Arc<T>` is never accessed.
    ///
    /// [into_raw]: Arc::into_raw
    /// [transmute]: core::mem::transmute
    /// [unsized coercion]: https://doc.rust-lang.org/reference/type-coercions.html#unsized-coercions
    ///
    /// # Examples
    ///
    /// ```
    /// use std::sync::Arc;
    ///
    /// let x = Arc::new("hello".to_owned());
    /// let x_ptr = Arc::into_raw(x);
    ///
    /// unsafe {
    ///     // Convert back to an `Arc` to prevent leak.
    ///     let x = Arc::from_raw(x_ptr);
    ///     assert_eq!(&*x, "hello");
    ///
    ///     // Further calls to `Arc::from_raw(x_ptr)` would be memory-unsafe.
    /// }
    ///
    /// // The memory was freed when `x` went out of scope above, so `x_ptr` is now dangling!
    /// ```
    ///
    /// Convert a slice back into its original array:
    ///
    /// ```
    /// use std::sync::Arc;
    ///
    /// let x: Arc<[u32]> = Arc::new([1, 2, 3]);
    /// let x_ptr: *const [u32] = Arc::into_raw(x);
    ///
    /// unsafe {
    ///     let x: Arc<[u32; 3]> = Arc::from_raw(x_ptr.cast::<[u32; 3]>());
    ///     assert_eq!(&*x, &[1, 2, 3]);
    /// }
    /// ```
    #[inline]
    #[stable(feature = "rc_raw", since = "1.17.0")]
    pub unsafe fn from_raw(ptr: *const T) -> Self {
        unsafe { Self { raw_rc: RawRc::from_raw(NonNull::new_unchecked(ptr.cast_mut())) } }
    }

    /// Consumes the `Arc`, returning the wrapped pointer.
    ///
    /// To avoid a memory leak the pointer must be converted back to an `Arc` using
    /// [`Arc::from_raw`].
    ///
    /// # Examples
    ///
    /// ```
    /// use std::sync::Arc;
    ///
    /// let x = Arc::new("hello".to_owned());
    /// let x_ptr = Arc::into_raw(x);
    /// assert_eq!(unsafe { &*x_ptr }, "hello");
    /// # // Prevent leaks for Miri.
    /// # drop(unsafe { Arc::from_raw(x_ptr) });
    /// ```
    #[must_use = "losing the pointer will leak memory"]
    #[stable(feature = "rc_raw", since = "1.17.0")]
    #[rustc_never_returns_null_ptr]
    pub fn into_raw(this: Self) -> *const T {
        Self::into_raw_rc(this).into_raw().as_ptr()
    }

    /// Increments the strong reference count on the `Arc<T>` associated with the
    /// provided pointer by one.
    ///
    /// # Safety
    ///
    /// The pointer must have been obtained through `Arc::into_raw` and must satisfy the
    /// same layout requirements specified in [`Arc::from_raw_in`][from_raw_in].
    /// The associated `Arc` instance must be valid (i.e. the strong count must be at
    /// least 1) for the duration of this method, and `ptr` must point to a block of memory
    /// allocated by the global allocator.
    ///
    /// [from_raw_in]: Arc::from_raw_in
    ///
    /// # Examples
    ///
    /// ```
    /// use std::sync::Arc;
    ///
    /// let five = Arc::new(5);
    ///
    /// unsafe {
    ///     let ptr = Arc::into_raw(five);
    ///     Arc::increment_strong_count(ptr);
    ///
    ///     // This assertion is deterministic because we haven't shared
    ///     // the `Arc` between threads.
    ///     let five = Arc::from_raw(ptr);
    ///     assert_eq!(2, Arc::strong_count(&five));
    /// #   // Prevent leaks for Miri.
    /// #   Arc::decrement_strong_count(ptr);
    /// }
    /// ```
    #[inline]
    #[stable(feature = "arc_mutate_strong_count", since = "1.51.0")]
    pub unsafe fn increment_strong_count(ptr: *const T) {
        unsafe {
            RawRc::<T, Global>::increment_strong_count::<RefCounter>(NonNull::new_unchecked(
                ptr.cast_mut(),
            ));
        }
    }

    /// Decrements the strong reference count on the `Arc<T>` associated with the
    /// provided pointer by one.
    ///
    /// # Safety
    ///
    /// The pointer must have been obtained through `Arc::into_raw` and must satisfy the
    /// same layout requirements specified in [`Arc::from_raw_in`][from_raw_in].
    /// The associated `Arc` instance must be valid (i.e. the strong count must be at
    /// least 1) when invoking this method, and `ptr` must point to a block of memory
    /// allocated by the global allocator. This method can be used to release the final
    /// `Arc` and backing storage, but **should not** be called after the final `Arc` has been
    /// released.
    ///
    /// [from_raw_in]: Arc::from_raw_in
    ///
    /// # Examples
    ///
    /// ```
    /// use std::sync::Arc;
    ///
    /// let five = Arc::new(5);
    ///
    /// unsafe {
    ///     let ptr = Arc::into_raw(five);
    ///     Arc::increment_strong_count(ptr);
    ///
    ///     // Those assertions are deterministic because we haven't shared
    ///     // the `Arc` between threads.
    ///     let five = Arc::from_raw(ptr);
    ///     assert_eq!(2, Arc::strong_count(&five));
    ///     Arc::decrement_strong_count(ptr);
    ///     assert_eq!(1, Arc::strong_count(&five));
    /// }
    /// ```
    #[inline]
    #[stable(feature = "arc_mutate_strong_count", since = "1.51.0")]
    pub unsafe fn decrement_strong_count(ptr: *const T) {
        unsafe {
            RawRc::<T, Global>::decrement_strong_count::<RefCounter>(NonNull::new_unchecked(
                ptr.cast_mut(),
            ));
        }
    }
}

impl<T: ?Sized, A: Allocator> Arc<T, A> {
    #[inline]
    fn into_raw_rc(this: Arc<T, A>) -> RawRc<T, A> {
        let this = ManuallyDrop::new(this);

        unsafe { ptr::read(&this.raw_rc) }
    }

    fn raw_strong_count(&self) -> &Atomic<usize> {
        unsafe { Atomic::<usize>::from_ptr(self.raw_rc.strong_count().get()) }
    }

    fn raw_weak_count(&self) -> &Atomic<usize> {
        unsafe { Atomic::<usize>::from_ptr(self.raw_rc.weak_count().get()) }
    }

    /// Returns a reference to the underlying allocator.
    ///
    /// Note: this is an associated function, which means that you have
    /// to call it as `Arc::allocator(&a)` instead of `a.allocator()`. This
    /// is so that there is no conflict with a method on the inner type.
    #[inline]
    #[unstable(feature = "allocator_api", issue = "32838")]
    pub fn allocator(this: &Self) -> &A {
        this.raw_rc.allocator()
    }

    /// Consumes the `Arc`, returning the wrapped pointer and allocator.
    ///
    /// To avoid a memory leak the pointer must be converted back to an `Arc` using
    /// [`Arc::from_raw_in`].
    ///
    /// # Examples
    ///
    /// ```
    /// #![feature(allocator_api)]
    /// use std::sync::Arc;
    /// use std::alloc::System;
    ///
    /// let x = Arc::new_in("hello".to_owned(), System);
    /// let (ptr, alloc) = Arc::into_raw_with_allocator(x);
    /// assert_eq!(unsafe { &*ptr }, "hello");
    /// let x = unsafe { Arc::from_raw_in(ptr, alloc) };
    /// assert_eq!(&*x, "hello");
    /// ```
    #[must_use = "losing the pointer will leak memory"]
    #[unstable(feature = "allocator_api", issue = "32838")]
    pub fn into_raw_with_allocator(this: Self) -> (*const T, A) {
        let (ptr, alloc) = Self::into_raw_rc(this).into_raw_parts();

        (ptr.as_ptr(), alloc)
    }

    /// Provides a raw pointer to the data.
    ///
    /// The counts are not affected in any way and the `Arc` is not consumed. The pointer is valid for
    /// as long as there are strong counts in the `Arc`.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::sync::Arc;
    ///
    /// let x = Arc::new("hello".to_owned());
    /// let y = Arc::clone(&x);
    /// let x_ptr = Arc::as_ptr(&x);
    /// assert_eq!(x_ptr, Arc::as_ptr(&y));
    /// assert_eq!(unsafe { &*x_ptr }, "hello");
    /// ```
    #[must_use]
    #[stable(feature = "rc_as_ptr", since = "1.45.0")]
    #[rustc_never_returns_null_ptr]
    pub fn as_ptr(this: &Self) -> *const T {
        this.raw_rc.as_ptr().as_ptr()
    }

    /// Constructs an `Arc<T, A>` from a raw pointer.
    ///
    /// The raw pointer must have been previously returned by a call to [`Arc<U,
    /// A>::into_raw`][into_raw] with the following requirements:
    ///
    /// * If `U` is sized, it must have the same size and alignment as `T`. This
    ///   is trivially true if `U` is `T`.
    /// * If `U` is unsized, its data pointer must have the same size and
    ///   alignment as `T`. This is trivially true if `Arc<U>` was constructed
    ///   through `Arc<T>` and then converted to `Arc<U>` through an [unsized
    ///   coercion].
    ///
    /// Note that if `U` or `U`'s data pointer is not `T` but has the same size
    /// and alignment, this is basically like transmuting references of
    /// different types. See [`mem::transmute`][transmute] for more information
    /// on what restrictions apply in this case.
    ///
    /// The raw pointer must point to a block of memory allocated by `alloc`
    ///
    /// The user of `from_raw` has to make sure a specific value of `T` is only
    /// dropped once.
    ///
    /// This function is unsafe because improper use may lead to memory unsafety,
    /// even if the returned `Arc<T>` is never accessed.
    ///
    /// [into_raw]: Arc::into_raw
    /// [transmute]: core::mem::transmute
    /// [unsized coercion]: https://doc.rust-lang.org/reference/type-coercions.html#unsized-coercions
    ///
    /// # Examples
    ///
    /// ```
    /// #![feature(allocator_api)]
    ///
    /// use std::sync::Arc;
    /// use std::alloc::System;
    ///
    /// let x = Arc::new_in("hello".to_owned(), System);
    /// let (x_ptr, alloc) = Arc::into_raw_with_allocator(x);
    ///
    /// unsafe {
    ///     // Convert back to an `Arc` to prevent leak.
    ///     let x = Arc::from_raw_in(x_ptr, System);
    ///     assert_eq!(&*x, "hello");
    ///
    ///     // Further calls to `Arc::from_raw(x_ptr)` would be memory-unsafe.
    /// }
    ///
    /// // The memory was freed when `x` went out of scope above, so `x_ptr` is now dangling!
    /// ```
    ///
    /// Convert a slice back into its original array:
    ///
    /// ```
    /// #![feature(allocator_api)]
    ///
    /// use std::sync::Arc;
    /// use std::alloc::System;
    ///
    /// let x: Arc<[u32], _> = Arc::new_in([1, 2, 3], System);
    /// let x_ptr: *const [u32] = Arc::into_raw_with_allocator(x).0;
    ///
    /// unsafe {
    ///     let x: Arc<[u32; 3], _> = Arc::from_raw_in(x_ptr.cast::<[u32; 3]>(), System);
    ///     assert_eq!(&*x, &[1, 2, 3]);
    /// }
    /// ```
    #[inline]
    #[unstable(feature = "allocator_api", issue = "32838")]
    pub unsafe fn from_raw_in(ptr: *const T, alloc: A) -> Self {
        Self {
            raw_rc: unsafe { RawRc::from_raw_parts(NonNull::new_unchecked(ptr.cast_mut()), alloc) },
        }
    }

    /// Creates a new [`Weak`] pointer to this allocation.
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
    #[must_use = "this returns a new `Weak` pointer, \
                  without modifying the original `Arc`"]
    #[stable(feature = "arc_weak", since = "1.4.0")]
    pub fn downgrade(this: &Self) -> Weak<T, A>
    where
        A: Clone,
    {
        Weak { raw_weak: unsafe { this.raw_rc.downgrade::<RefCounter>() } }
    }

    /// Gets the number of [`Weak`] pointers to this allocation.
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
    #[must_use]
    #[stable(feature = "arc_counts", since = "1.15.0")]
    pub fn weak_count(this: &Self) -> usize {
        let cnt = this.raw_weak_count().load(Relaxed);
        // If the weak count is currently locked, the value of the
        // count was 0 just before taking the lock.
        if cnt == usize::MAX { 0 } else { cnt - 1 }
    }

    /// Gets the number of strong (`Arc`) pointers to this allocation.
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
    /// let _also_five = Arc::clone(&five);
    ///
    /// // This assertion is deterministic because we haven't shared
    /// // the `Arc` between threads.
    /// assert_eq!(2, Arc::strong_count(&five));
    /// ```
    #[inline]
    #[must_use]
    #[stable(feature = "arc_counts", since = "1.15.0")]
    pub fn strong_count(this: &Self) -> usize {
        this.raw_strong_count().load(Relaxed)
    }

    /// Increments the strong reference count on the `Arc<T>` associated with the
    /// provided pointer by one.
    ///
    /// # Safety
    ///
    /// The pointer must have been obtained through `Arc::into_raw` and must satisfy the
    /// same layout requirements specified in [`Arc::from_raw_in`][from_raw_in].
    /// The associated `Arc` instance must be valid (i.e. the strong count must be at
    /// least 1) for the duration of this method, and `ptr` must point to a block of memory
    /// allocated by `alloc`.
    ///
    /// [from_raw_in]: Arc::from_raw_in
    ///
    /// # Examples
    ///
    /// ```
    /// #![feature(allocator_api)]
    ///
    /// use std::sync::Arc;
    /// use std::alloc::System;
    ///
    /// let five = Arc::new_in(5, System);
    ///
    /// unsafe {
    ///     let (ptr, _alloc) = Arc::into_raw_with_allocator(five);
    ///     Arc::increment_strong_count_in(ptr, System);
    ///
    ///     // This assertion is deterministic because we haven't shared
    ///     // the `Arc` between threads.
    ///     let five = Arc::from_raw_in(ptr, System);
    ///     assert_eq!(2, Arc::strong_count(&five));
    /// #   // Prevent leaks for Miri.
    /// #   Arc::decrement_strong_count_in(ptr, System);
    /// }
    /// ```
    #[inline]
    #[unstable(feature = "allocator_api", issue = "32838")]
    pub unsafe fn increment_strong_count_in(ptr: *const T, alloc: A)
    where
        A: Clone,
    {
        unsafe {
            RawRc::<T, A>::increment_strong_count::<RefCounter>(NonNull::new_unchecked(
                ptr.cast_mut(),
            ));
        }

        drop(alloc);
    }

    /// Decrements the strong reference count on the `Arc<T>` associated with the
    /// provided pointer by one.
    ///
    /// # Safety
    ///
    /// The pointer must have been obtained through `Arc::into_raw` and must satisfy the
    /// same layout requirements specified in [`Arc::from_raw_in`][from_raw_in].
    /// The associated `Arc` instance must be valid (i.e. the strong count must be at
    /// least 1) when invoking this method, and `ptr` must point to a block of memory
    /// allocated by `alloc`. This method can be used to release the final
    /// `Arc` and backing storage, but **should not** be called after the final `Arc` has been
    /// released.
    ///
    /// [from_raw_in]: Arc::from_raw_in
    ///
    /// # Examples
    ///
    /// ```
    /// #![feature(allocator_api)]
    ///
    /// use std::sync::Arc;
    /// use std::alloc::System;
    ///
    /// let five = Arc::new_in(5, System);
    ///
    /// unsafe {
    ///     let (ptr, _alloc) = Arc::into_raw_with_allocator(five);
    ///     Arc::increment_strong_count_in(ptr, System);
    ///
    ///     // Those assertions are deterministic because we haven't shared
    ///     // the `Arc` between threads.
    ///     let five = Arc::from_raw_in(ptr, System);
    ///     assert_eq!(2, Arc::strong_count(&five));
    ///     Arc::decrement_strong_count_in(ptr, System);
    ///     assert_eq!(1, Arc::strong_count(&five));
    /// }
    /// ```
    #[inline]
    #[unstable(feature = "allocator_api", issue = "32838")]
    pub unsafe fn decrement_strong_count_in(ptr: *const T, alloc: A) {
        unsafe {
            RawRc::<T, A>::decrement_strong_count_in::<RefCounter>(
                NonNull::new_unchecked(ptr.cast_mut()),
                alloc,
            );
        }
    }

    /// Returns `true` if the two `Arc`s point to the same allocation in a vein similar to
    /// [`ptr::eq`]. This function ignores the metadata of  `dyn Trait` pointers.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::sync::Arc;
    ///
    /// let five = Arc::new(5);
    /// let same_five = Arc::clone(&five);
    /// let other_five = Arc::new(5);
    ///
    /// assert!(Arc::ptr_eq(&five, &same_five));
    /// assert!(!Arc::ptr_eq(&five, &other_five));
    /// ```
    ///
    /// [`ptr::eq`]: core::ptr::eq "ptr::eq"
    #[inline]
    #[must_use]
    #[stable(feature = "ptr_eq", since = "1.17.0")]
    pub fn ptr_eq(this: &Self, other: &Self) -> bool {
        RawRc::ptr_eq(&this.raw_rc, &other.raw_rc)
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<T: ?Sized, A: Allocator + Clone> Clone for Arc<T, A> {
    /// Makes a clone of the `Arc` pointer.
    ///
    /// This creates another pointer to the same allocation, increasing the
    /// strong reference count.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::sync::Arc;
    ///
    /// let five = Arc::new(5);
    ///
    /// let _ = Arc::clone(&five);
    /// ```
    #[inline]
    fn clone(&self) -> Arc<T, A> {
        Self { raw_rc: unsafe { self.raw_rc.clone::<RefCounter>() } }
    }
}

#[unstable(feature = "ergonomic_clones", issue = "132290")]
impl<T: ?Sized, A: Allocator + Clone> UseCloned for Arc<T, A> {}

#[stable(feature = "rust1", since = "1.0.0")]
impl<T: ?Sized, A: Allocator> Deref for Arc<T, A> {
    type Target = T;

    #[inline]
    fn deref(&self) -> &T {
        self.raw_rc.as_ref()
    }
}

#[unstable(feature = "pin_coerce_unsized_trait", issue = "150112")]
unsafe impl<T: ?Sized, A: Allocator> PinCoerceUnsized for Arc<T, A> {}

#[unstable(feature = "pin_coerce_unsized_trait", issue = "150112")]
unsafe impl<T: ?Sized, A: Allocator> PinCoerceUnsized for Weak<T, A> {}

#[unstable(feature = "deref_pure_trait", issue = "87121")]
unsafe impl<T: ?Sized, A: Allocator> DerefPure for Arc<T, A> {}

#[unstable(feature = "legacy_receiver_trait", issue = "none")]
impl<T: ?Sized> LegacyReceiver for Arc<T> {}

#[cfg(not(no_global_oom_handling))]
impl<T: ?Sized + CloneToUninit, A: Allocator + Clone> Arc<T, A> {
    /// Makes a mutable reference into the given `Arc`.
    ///
    /// If there are other `Arc` pointers to the same allocation, then `make_mut` will
    /// [`clone`] the inner value to a new allocation to ensure unique ownership.  This is also
    /// referred to as clone-on-write.
    ///
    /// However, if there are no other `Arc` pointers to this allocation, but some [`Weak`]
    /// pointers, then the [`Weak`] pointers will be dissociated and the inner value will not
    /// be cloned.
    ///
    /// See also [`get_mut`], which will fail rather than cloning the inner value
    /// or dissociating [`Weak`] pointers.
    ///
    /// [`clone`]: Clone::clone
    /// [`get_mut`]: Arc::get_mut
    ///
    /// # Examples
    ///
    /// ```
    /// use std::sync::Arc;
    ///
    /// let mut data = Arc::new(5);
    ///
    /// *Arc::make_mut(&mut data) += 1;         // Won't clone anything
    /// let mut other_data = Arc::clone(&data); // Won't clone inner data
    /// *Arc::make_mut(&mut data) += 1;         // Clones inner data
    /// *Arc::make_mut(&mut data) += 1;         // Won't clone anything
    /// *Arc::make_mut(&mut other_data) *= 2;   // Won't clone anything
    ///
    /// // Now `data` and `other_data` point to different allocations.
    /// assert_eq!(*data, 8);
    /// assert_eq!(*other_data, 12);
    /// ```
    ///
    /// [`Weak`] pointers will be dissociated:
    ///
    /// ```
    /// use std::sync::Arc;
    ///
    /// let mut data = Arc::new(75);
    /// let weak = Arc::downgrade(&data);
    ///
    /// assert!(75 == *data);
    /// assert!(75 == *weak.upgrade().unwrap());
    ///
    /// *Arc::make_mut(&mut data) += 1;
    ///
    /// assert!(76 == *data);
    /// assert!(weak.upgrade().is_none());
    /// ```
    #[inline]
    #[stable(feature = "arc_unique", since = "1.4.0")]
    pub fn make_mut(this: &mut Self) -> &mut T {
        unsafe { this.raw_rc.make_mut::<RefCounter>() }
    }
}

impl<T: Clone, A: Allocator> Arc<T, A> {
    /// If we have the only reference to `T` then unwrap it. Otherwise, clone `T` and return the
    /// clone.
    ///
    /// Assuming `arc_t` is of type `Arc<T>`, this function is functionally equivalent to
    /// `(*arc_t).clone()`, but will avoid cloning the inner value where possible.
    ///
    /// # Examples
    ///
    /// ```
    /// # use std::{ptr, sync::Arc};
    /// let inner = String::from("test");
    /// let ptr = inner.as_ptr();
    ///
    /// let arc = Arc::new(inner);
    /// let inner = Arc::unwrap_or_clone(arc);
    /// // The inner value was not cloned
    /// assert!(ptr::eq(ptr, inner.as_ptr()));
    ///
    /// let arc = Arc::new(inner);
    /// let arc2 = arc.clone();
    /// let inner = Arc::unwrap_or_clone(arc);
    /// // Because there were 2 references, we had to clone the inner value.
    /// assert!(!ptr::eq(ptr, inner.as_ptr()));
    /// // `arc2` is the last reference, so when we unwrap it we get back
    /// // the original `String`.
    /// let inner = Arc::unwrap_or_clone(arc2);
    /// assert!(ptr::eq(ptr, inner.as_ptr()));
    /// ```
    #[inline]
    #[stable(feature = "arc_unwrap_or_clone", since = "1.76.0")]
    pub fn unwrap_or_clone(this: Self) -> T {
        let raw_rc = Self::into_raw_rc(this);

        unsafe { raw_rc.unwrap_or_clone::<RefCounter>() }
    }
}

impl<T: ?Sized, A: Allocator> Arc<T, A> {
    /// Returns a mutable reference into the given `Arc`, if there are
    /// no other `Arc` or [`Weak`] pointers to the same allocation.
    ///
    /// Returns [`None`] otherwise, because it is not safe to
    /// mutate a shared value.
    ///
    /// See also [`make_mut`][make_mut], which will [`clone`][clone]
    /// the inner value when there are other `Arc` pointers.
    ///
    /// [make_mut]: Arc::make_mut
    /// [clone]: Clone::clone
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
    /// let _y = Arc::clone(&x);
    /// assert!(Arc::get_mut(&mut x).is_none());
    /// ```
    #[inline]
    #[stable(feature = "arc_unique", since = "1.4.0")]
    pub fn get_mut(this: &mut Self) -> Option<&mut T> {
        unsafe { this.raw_rc.get_mut::<RefCounter>() }
    }

    /// Returns a mutable reference into the given `Arc`,
    /// without any check.
    ///
    /// See also [`get_mut`], which is safe and does appropriate checks.
    ///
    /// [`get_mut`]: Arc::get_mut
    ///
    /// # Safety
    ///
    /// If any other `Arc` or [`Weak`] pointers to the same allocation exist, then
    /// they must not be dereferenced or have active borrows for the duration
    /// of the returned borrow, and their inner type must be exactly the same as the
    /// inner type of this Arc (including lifetimes). This is trivially the case if no
    /// such pointers exist, for example immediately after `Arc::new`.
    ///
    /// # Examples
    ///
    /// ```
    /// #![feature(get_mut_unchecked)]
    ///
    /// use std::sync::Arc;
    ///
    /// let mut x = Arc::new(String::new());
    /// unsafe {
    ///     Arc::get_mut_unchecked(&mut x).push_str("foo")
    /// }
    /// assert_eq!(*x, "foo");
    /// ```
    /// Other `Arc` pointers to the same allocation must be to the same type.
    /// ```no_run
    /// #![feature(get_mut_unchecked)]
    ///
    /// use std::sync::Arc;
    ///
    /// let x: Arc<str> = Arc::from("Hello, world!");
    /// let mut y: Arc<[u8]> = x.clone().into();
    /// unsafe {
    ///     // this is Undefined Behavior, because x's inner type is str, not [u8]
    ///     Arc::get_mut_unchecked(&mut y).fill(0xff); // 0xff is invalid in UTF-8
    /// }
    /// println!("{}", &*x); // Invalid UTF-8 in a str
    /// ```
    /// Other `Arc` pointers to the same allocation must be to the exact same type, including lifetimes.
    /// ```no_run
    /// #![feature(get_mut_unchecked)]
    ///
    /// use std::sync::Arc;
    ///
    /// let x: Arc<&str> = Arc::new("Hello, world!");
    /// {
    ///     let s = String::from("Oh, no!");
    ///     let mut y: Arc<&str> = x.clone();
    ///     unsafe {
    ///         // this is Undefined Behavior, because x's inner type
    ///         // is &'long str, not &'short str
    ///         *Arc::get_mut_unchecked(&mut y) = &s;
    ///     }
    /// }
    /// println!("{}", &*x); // Use-after-free
    /// ```
    #[inline]
    #[unstable(feature = "get_mut_unchecked", issue = "63292")]
    pub unsafe fn get_mut_unchecked(this: &mut Self) -> &mut T {
        unsafe { this.raw_rc.get_mut_unchecked() }
    }

    /// Determine whether this is the unique reference to the underlying data.
    ///
    /// Returns `true` if there are no other `Arc` or [`Weak`] pointers to the same allocation;
    /// returns `false` otherwise.
    ///
    /// If this function returns `true`, then is guaranteed to be safe to call [`get_mut_unchecked`]
    /// on this `Arc`, so long as no clones occur in between.
    ///
    /// # Examples
    ///
    /// ```
    /// #![feature(arc_is_unique)]
    ///
    /// use std::sync::Arc;
    ///
    /// let x = Arc::new(3);
    /// assert!(Arc::is_unique(&x));
    ///
    /// let y = Arc::clone(&x);
    /// assert!(!Arc::is_unique(&x));
    /// drop(y);
    ///
    /// // Weak references also count, because they could be upgraded at any time.
    /// let z = Arc::downgrade(&x);
    /// assert!(!Arc::is_unique(&x));
    /// ```
    ///
    /// # Pointer invalidation
    ///
    /// This function will always return the same value as `Arc::get_mut(arc).is_some()`. However,
    /// unlike that operation it does not produce any mutable references to the underlying data,
    /// meaning no pointers to the data inside the `Arc` are invalidated by the call. Thus, the
    /// following code is valid, even though it would be UB if it used `Arc::get_mut`:
    ///
    /// ```
    /// #![feature(arc_is_unique)]
    ///
    /// use std::sync::Arc;
    ///
    /// let arc = Arc::new(5);
    /// let pointer: *const i32 = &*arc;
    /// assert!(Arc::is_unique(&arc));
    /// assert_eq!(unsafe { *pointer }, 5);
    /// ```
    ///
    /// # Atomic orderings
    ///
    /// Concurrent drops to other `Arc` pointers to the same allocation will synchronize with this
    /// call - that is, this call performs an `Acquire` operation on the underlying strong and weak
    /// ref counts. This ensures that calling `get_mut_unchecked` is safe.
    ///
    /// Note that this operation requires locking the weak ref count, so concurrent calls to
    /// `downgrade` may spin-loop for a short period of time.
    ///
    /// [`get_mut_unchecked`]: Self::get_mut_unchecked
    #[inline]
    #[unstable(feature = "arc_is_unique", issue = "138938")]
    pub fn is_unique(this: &Self) -> bool {
        unsafe { this.raw_rc.is_unique::<RefCounter>() }
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
unsafe impl<#[may_dangle] T: ?Sized, A: Allocator> Drop for Arc<T, A> {
    /// Drops the `Arc`.
    ///
    /// This will decrement the strong reference count. If the strong reference
    /// count reaches zero then the only other references (if any) are
    /// [`Weak`], so we `drop` the inner value.
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
    /// let foo2 = Arc::clone(&foo);
    ///
    /// drop(foo);    // Doesn't print anything
    /// drop(foo2);   // Prints "dropped!"
    /// ```
    #[inline]
    fn drop(&mut self) {
        unsafe { self.raw_rc.drop::<RefCounter>() };
    }
}

impl<A: Allocator> Arc<dyn Any + Send + Sync, A> {
    /// Attempts to downcast the `Arc<dyn Any + Send + Sync>` to a concrete type.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::any::Any;
    /// use std::sync::Arc;
    ///
    /// fn print_if_string(value: Arc<dyn Any + Send + Sync>) {
    ///     if let Ok(string) = value.downcast::<String>() {
    ///         println!("String ({}): {}", string.len(), string);
    ///     }
    /// }
    ///
    /// let my_string = "Hello World".to_string();
    /// print_if_string(Arc::new(my_string));
    /// print_if_string(Arc::new(0i8));
    /// ```
    #[inline]
    #[stable(feature = "rc_downcast", since = "1.29.0")]
    pub fn downcast<T>(self) -> Result<Arc<T, A>, Self>
    where
        T: Any + Send + Sync,
    {
        match Self::into_raw_rc(self).downcast::<T>() {
            Ok(raw_rc) => Ok(Arc { raw_rc }),
            Err(raw_rc) => Err(Self { raw_rc }),
        }
    }

    /// Downcasts the `Arc<dyn Any + Send + Sync>` to a concrete type.
    ///
    /// For a safe alternative see [`downcast`].
    ///
    /// # Examples
    ///
    /// ```
    /// #![feature(downcast_unchecked)]
    ///
    /// use std::any::Any;
    /// use std::sync::Arc;
    ///
    /// let x: Arc<dyn Any + Send + Sync> = Arc::new(1_usize);
    ///
    /// unsafe {
    ///     assert_eq!(*x.downcast_unchecked::<usize>(), 1);
    /// }
    /// ```
    ///
    /// # Safety
    ///
    /// The contained value must be of type `T`. Calling this method
    /// with the incorrect type is *undefined behavior*.
    ///
    ///
    /// [`downcast`]: Self::downcast
    #[inline]
    #[unstable(feature = "downcast_unchecked", issue = "90850")]
    pub unsafe fn downcast_unchecked<T>(self) -> Arc<T, A>
    where
        T: Any + Send + Sync,
    {
        let raw_rc = Self::into_raw_rc(self);
        let raw_rc = unsafe { raw_rc.downcast_unchecked::<T>() };

        Arc { raw_rc }
    }
}

impl<T> Weak<T> {
    /// Constructs a new `Weak<T>`, without allocating any memory.
    /// Calling [`upgrade`] on the return value always gives [`None`].
    ///
    /// [`upgrade`]: Weak::upgrade
    ///
    /// # Examples
    ///
    /// ```
    /// use std::sync::Weak;
    ///
    /// let empty: Weak<i64> = Weak::new();
    /// assert!(empty.upgrade().is_none());
    /// ```
    #[inline]
    #[stable(feature = "downgraded_weak", since = "1.10.0")]
    #[rustc_const_stable(feature = "const_weak_new", since = "1.73.0")]
    #[must_use]
    pub const fn new() -> Weak<T> {
        Self { raw_weak: RawWeak::new_dangling_in(Global) }
    }
}

impl<T, A: Allocator> Weak<T, A> {
    /// Constructs a new `Weak<T, A>`, without allocating any memory, technically in the provided
    /// allocator.
    /// Calling [`upgrade`] on the return value always gives [`None`].
    ///
    /// [`upgrade`]: Weak::upgrade
    ///
    /// # Examples
    ///
    /// ```
    /// #![feature(allocator_api)]
    ///
    /// use std::sync::Weak;
    /// use std::alloc::System;
    ///
    /// let empty: Weak<i64, _> = Weak::new_in(System);
    /// assert!(empty.upgrade().is_none());
    /// ```
    #[inline]
    #[unstable(feature = "allocator_api", issue = "32838")]
    pub fn new_in(alloc: A) -> Weak<T, A> {
        Self { raw_weak: RawWeak::new_dangling_in(alloc) }
    }
}

impl<T: ?Sized> Weak<T> {
    /// Converts a raw pointer previously created by [`into_raw`] back into `Weak<T>`.
    ///
    /// This can be used to safely get a strong reference (by calling [`upgrade`]
    /// later) or to deallocate the weak count by dropping the `Weak<T>`.
    ///
    /// It takes ownership of one weak reference (with the exception of pointers created by [`new`],
    /// as these don't own anything; the method still works on them).
    ///
    /// # Safety
    ///
    /// The pointer must have originated from the [`into_raw`] and must still own its potential
    /// weak reference, and must point to a block of memory allocated by global allocator.
    ///
    /// It is allowed for the strong count to be 0 at the time of calling this. Nevertheless, this
    /// takes ownership of one weak reference currently represented as a raw pointer (the weak
    /// count is not modified by this operation) and therefore it must be paired with a previous
    /// call to [`into_raw`].
    /// # Examples
    ///
    /// ```
    /// use std::sync::{Arc, Weak};
    ///
    /// let strong = Arc::new("hello".to_owned());
    ///
    /// let raw_1 = Arc::downgrade(&strong).into_raw();
    /// let raw_2 = Arc::downgrade(&strong).into_raw();
    ///
    /// assert_eq!(2, Arc::weak_count(&strong));
    ///
    /// assert_eq!("hello", &*unsafe { Weak::from_raw(raw_1) }.upgrade().unwrap());
    /// assert_eq!(1, Arc::weak_count(&strong));
    ///
    /// drop(strong);
    ///
    /// // Decrement the last weak count.
    /// assert!(unsafe { Weak::from_raw(raw_2) }.upgrade().is_none());
    /// ```
    ///
    /// [`new`]: Weak::new
    /// [`into_raw`]: Weak::into_raw
    /// [`upgrade`]: Weak::upgrade
    #[inline]
    #[stable(feature = "weak_into_raw", since = "1.45.0")]
    pub unsafe fn from_raw(ptr: *const T) -> Self {
        Self { raw_weak: unsafe { RawWeak::from_raw(NonNull::new_unchecked(ptr.cast_mut())) } }
    }

    /// Consumes the `Weak<T>` and turns it into a raw pointer.
    ///
    /// This converts the weak pointer into a raw pointer, while still preserving the ownership of
    /// one weak reference (the weak count is not modified by this operation). It can be turned
    /// back into the `Weak<T>` with [`from_raw`].
    ///
    /// The same restrictions of accessing the target of the pointer as with
    /// [`as_ptr`] apply.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::sync::{Arc, Weak};
    ///
    /// let strong = Arc::new("hello".to_owned());
    /// let weak = Arc::downgrade(&strong);
    /// let raw = weak.into_raw();
    ///
    /// assert_eq!(1, Arc::weak_count(&strong));
    /// assert_eq!("hello", unsafe { &*raw });
    ///
    /// drop(unsafe { Weak::from_raw(raw) });
    /// assert_eq!(0, Arc::weak_count(&strong));
    /// ```
    ///
    /// [`from_raw`]: Weak::from_raw
    /// [`as_ptr`]: Weak::as_ptr
    #[must_use = "losing the pointer will leak memory"]
    #[stable(feature = "weak_into_raw", since = "1.45.0")]
    pub fn into_raw(self) -> *const T {
        self.into_raw_weak().into_raw().as_ptr()
    }
}

impl<T: ?Sized, A: Allocator> Weak<T, A> {
    #[cfg(not(no_global_oom_handling))]
    #[inline]
    fn ref_from_raw_weak(raw_weak: &RawWeak<T, A>) -> &Self {
        // SAFETY: This is safe because `Weak` has transparent representation of `RawWeak`.
        unsafe { mem::transmute(raw_weak) }
    }

    #[inline]
    fn into_raw_weak(self) -> RawWeak<T, A> {
        let this = ManuallyDrop::new(self);

        unsafe { ptr::read(&this.raw_weak) }
    }

    fn raw_strong_count(&self) -> Option<&Atomic<usize>> {
        self.raw_weak
            .strong_count()
            .map(|strong_count| unsafe { Atomic::<usize>::from_ptr(strong_count.get()) })
    }

    /// Returns a reference to the underlying allocator.
    #[inline]
    #[unstable(feature = "allocator_api", issue = "32838")]
    pub fn allocator(&self) -> &A {
        self.raw_weak.allocator()
    }

    /// Returns a raw pointer to the object `T` pointed to by this `Weak<T>`.
    ///
    /// The pointer is valid only if there are some strong references. The pointer may be dangling,
    /// unaligned or even [`null`] otherwise.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::sync::Arc;
    /// use std::ptr;
    ///
    /// let strong = Arc::new("hello".to_owned());
    /// let weak = Arc::downgrade(&strong);
    /// // Both point to the same object
    /// assert!(ptr::eq(&*strong, weak.as_ptr()));
    /// // The strong here keeps it alive, so we can still access the object.
    /// assert_eq!("hello", unsafe { &*weak.as_ptr() });
    ///
    /// drop(strong);
    /// // But not any more. We can do weak.as_ptr(), but accessing the pointer would lead to
    /// // undefined behavior.
    /// // assert_eq!("hello", unsafe { &*weak.as_ptr() });
    /// ```
    ///
    /// [`null`]: core::ptr::null "ptr::null"
    #[must_use]
    #[stable(feature = "weak_into_raw", since = "1.45.0")]
    pub fn as_ptr(&self) -> *const T {
        self.raw_weak.as_ptr().as_ptr()
    }

    /// Consumes the `Weak<T>`, returning the wrapped pointer and allocator.
    ///
    /// This converts the weak pointer into a raw pointer, while still preserving the ownership of
    /// one weak reference (the weak count is not modified by this operation). It can be turned
    /// back into the `Weak<T>` with [`from_raw_in`].
    ///
    /// The same restrictions of accessing the target of the pointer as with
    /// [`as_ptr`] apply.
    ///
    /// # Examples
    ///
    /// ```
    /// #![feature(allocator_api)]
    /// use std::sync::{Arc, Weak};
    /// use std::alloc::System;
    ///
    /// let strong = Arc::new_in("hello".to_owned(), System);
    /// let weak = Arc::downgrade(&strong);
    /// let (raw, alloc) = weak.into_raw_with_allocator();
    ///
    /// assert_eq!(1, Arc::weak_count(&strong));
    /// assert_eq!("hello", unsafe { &*raw });
    ///
    /// drop(unsafe { Weak::from_raw_in(raw, alloc) });
    /// assert_eq!(0, Arc::weak_count(&strong));
    /// ```
    ///
    /// [`from_raw_in`]: Weak::from_raw_in
    /// [`as_ptr`]: Weak::as_ptr
    #[must_use = "losing the pointer will leak memory"]
    #[unstable(feature = "allocator_api", issue = "32838")]
    pub fn into_raw_with_allocator(self) -> (*const T, A) {
        let (ptr, alloc) = self.into_raw_weak().into_raw_parts();

        (ptr.as_ptr(), alloc)
    }

    /// Converts a raw pointer previously created by [`into_raw`] back into `Weak<T>` in the provided
    /// allocator.
    ///
    /// This can be used to safely get a strong reference (by calling [`upgrade`]
    /// later) or to deallocate the weak count by dropping the `Weak<T>`.
    ///
    /// It takes ownership of one weak reference (with the exception of pointers created by [`new`],
    /// as these don't own anything; the method still works on them).
    ///
    /// # Safety
    ///
    /// The pointer must have originated from the [`into_raw`] and must still own its potential
    /// weak reference, and must point to a block of memory allocated by `alloc`.
    ///
    /// It is allowed for the strong count to be 0 at the time of calling this. Nevertheless, this
    /// takes ownership of one weak reference currently represented as a raw pointer (the weak
    /// count is not modified by this operation) and therefore it must be paired with a previous
    /// call to [`into_raw`].
    /// # Examples
    ///
    /// ```
    /// use std::sync::{Arc, Weak};
    ///
    /// let strong = Arc::new("hello".to_owned());
    ///
    /// let raw_1 = Arc::downgrade(&strong).into_raw();
    /// let raw_2 = Arc::downgrade(&strong).into_raw();
    ///
    /// assert_eq!(2, Arc::weak_count(&strong));
    ///
    /// assert_eq!("hello", &*unsafe { Weak::from_raw(raw_1) }.upgrade().unwrap());
    /// assert_eq!(1, Arc::weak_count(&strong));
    ///
    /// drop(strong);
    ///
    /// // Decrement the last weak count.
    /// assert!(unsafe { Weak::from_raw(raw_2) }.upgrade().is_none());
    /// ```
    ///
    /// [`new`]: Weak::new
    /// [`into_raw`]: Weak::into_raw
    /// [`upgrade`]: Weak::upgrade
    #[inline]
    #[unstable(feature = "allocator_api", issue = "32838")]
    pub unsafe fn from_raw_in(ptr: *const T, alloc: A) -> Self {
        Self {
            raw_weak: unsafe {
                RawWeak::from_raw_parts(NonNull::new_unchecked(ptr.cast_mut()), alloc)
            },
        }
    }
}

impl<T: ?Sized, A: Allocator> Weak<T, A> {
    /// Attempts to upgrade the `Weak` pointer to an [`Arc`], delaying
    /// dropping of the inner value if successful.
    ///
    /// Returns [`None`] if the inner value has since been dropped.
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
    #[must_use = "this returns a new `Arc`, \
                  without modifying the original weak pointer"]
    #[stable(feature = "arc_weak", since = "1.4.0")]
    pub fn upgrade(&self) -> Option<Arc<T, A>>
    where
        A: Clone,
    {
        unsafe { self.raw_weak.upgrade::<RefCounter>() }.map(|raw_rc| Arc { raw_rc })
    }

    /// Gets the number of strong (`Arc`) pointers pointing to this allocation.
    ///
    /// If `self` was created using [`Weak::new`], this will return 0.
    #[must_use]
    #[stable(feature = "weak_counts", since = "1.41.0")]
    pub fn strong_count(&self) -> usize {
        self.raw_strong_count().map_or(0, |strong_count| strong_count.load(Relaxed))
    }

    /// Gets an approximation of the number of `Weak` pointers pointing to this
    /// allocation.
    ///
    /// If `self` was created using [`Weak::new`], or if there are no remaining
    /// strong pointers, this will return 0.
    ///
    /// # Accuracy
    ///
    /// Due to implementation details, the returned value can be off by 1 in
    /// either direction when other threads are manipulating any `Arc`s or
    /// `Weak`s pointing to the same allocation.
    #[must_use]
    #[stable(feature = "weak_counts", since = "1.41.0")]
    pub fn weak_count(&self) -> usize {
        if let Some(ref_counts) = self.raw_weak.ref_counts() {
            let weak = unsafe { Atomic::<usize>::from_ptr(ref_counts.weak.get()) }.load(Acquire);
            let strong =
                unsafe { Atomic::<usize>::from_ptr(ref_counts.strong.get()) }.load(Relaxed);
            if strong == 0 {
                0
            } else {
                // Since we observed that there was at least one strong pointer
                // after reading the weak count, we know that the implicit weak
                // reference (present whenever any strong references are alive)
                // was still around when we observed the weak count, and can
                // therefore safely subtract it.
                weak - 1
            }
        } else {
            0
        }
    }

    /// Returns `true` if the two `Weak`s point to the same allocation similar to [`ptr::eq`], or if
    /// both don't point to any allocation (because they were created with `Weak::new()`). However,
    /// this function ignores the metadata of  `dyn Trait` pointers.
    ///
    /// # Notes
    ///
    /// Since this compares pointers it means that `Weak::new()` will equal each
    /// other, even though they don't point to any allocation.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::sync::Arc;
    ///
    /// let first_rc = Arc::new(5);
    /// let first = Arc::downgrade(&first_rc);
    /// let second = Arc::downgrade(&first_rc);
    ///
    /// assert!(first.ptr_eq(&second));
    ///
    /// let third_rc = Arc::new(5);
    /// let third = Arc::downgrade(&third_rc);
    ///
    /// assert!(!first.ptr_eq(&third));
    /// ```
    ///
    /// Comparing `Weak::new`.
    ///
    /// ```
    /// use std::sync::{Arc, Weak};
    ///
    /// let first = Weak::new();
    /// let second = Weak::new();
    /// assert!(first.ptr_eq(&second));
    ///
    /// let third_rc = Arc::new(());
    /// let third = Arc::downgrade(&third_rc);
    /// assert!(!first.ptr_eq(&third));
    /// ```
    ///
    /// [`ptr::eq`]: core::ptr::eq "ptr::eq"
    #[inline]
    #[must_use]
    #[stable(feature = "weak_ptr_eq", since = "1.39.0")]
    pub fn ptr_eq(&self, other: &Self) -> bool {
        RawWeak::ptr_eq(&self.raw_weak, &other.raw_weak)
    }
}

#[stable(feature = "arc_weak", since = "1.4.0")]
impl<T: ?Sized, A: Allocator + Clone> Clone for Weak<T, A> {
    /// Makes a clone of the `Weak` pointer that points to the same allocation.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::sync::{Arc, Weak};
    ///
    /// let weak_five = Arc::downgrade(&Arc::new(5));
    ///
    /// let _ = Weak::clone(&weak_five);
    /// ```
    #[inline]
    fn clone(&self) -> Weak<T, A> {
        Self { raw_weak: unsafe { self.raw_weak.clone::<RefCounter>() } }
    }
}

#[unstable(feature = "ergonomic_clones", issue = "132290")]
impl<T: ?Sized, A: Allocator + Clone> UseCloned for Weak<T, A> {}

#[stable(feature = "downgraded_weak", since = "1.10.0")]
impl<T> Default for Weak<T> {
    /// Constructs a new `Weak<T>`, without allocating memory.
    /// Calling [`upgrade`] on the return value always
    /// gives [`None`].
    ///
    /// [`upgrade`]: Weak::upgrade
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
        Self { raw_weak: RawWeak::default() }
    }
}

#[stable(feature = "arc_weak", since = "1.4.0")]
unsafe impl<#[may_dangle] T: ?Sized, A: Allocator> Drop for Weak<T, A> {
    /// Drops the `Weak` pointer.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::sync::{Arc, Weak};
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
    /// let other_weak_foo = Weak::clone(&weak_foo);
    ///
    /// drop(weak_foo);   // Doesn't print anything
    /// drop(foo);        // Prints "dropped!"
    ///
    /// assert!(other_weak_foo.upgrade().is_none());
    /// ```
    fn drop(&mut self) {
        unsafe { self.raw_weak.drop::<RefCounter>() };
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<T: ?Sized + PartialEq, A: Allocator> PartialEq for Arc<T, A> {
    /// Equality for two `Arc`s.
    ///
    /// Two `Arc`s are equal if their inner values are equal, even if they are
    /// stored in different allocation.
    ///
    /// If `T` also implements `Eq` (implying reflexivity of equality),
    /// two `Arc`s that point to the same allocation are always equal.
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
    #[inline]
    fn eq(&self, other: &Arc<T, A>) -> bool {
        RawRc::eq(&self.raw_rc, &other.raw_rc)
    }

    /// Inequality for two `Arc`s.
    ///
    /// Two `Arc`s are not equal if their inner values are not equal.
    ///
    /// If `T` also implements `Eq` (implying reflexivity of equality),
    /// two `Arc`s that point to the same value are always equal.
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
    #[inline]
    fn ne(&self, other: &Arc<T, A>) -> bool {
        RawRc::ne(&self.raw_rc, &other.raw_rc)
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<T: ?Sized + PartialOrd, A: Allocator> PartialOrd for Arc<T, A> {
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
    fn partial_cmp(&self, other: &Arc<T, A>) -> Option<Ordering> {
        RawRc::partial_cmp(&self.raw_rc, &other.raw_rc)
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
    fn lt(&self, other: &Arc<T, A>) -> bool {
        RawRc::lt(&self.raw_rc, &other.raw_rc)
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
    fn le(&self, other: &Arc<T, A>) -> bool {
        RawRc::le(&self.raw_rc, &other.raw_rc)
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
    fn gt(&self, other: &Arc<T, A>) -> bool {
        RawRc::gt(&self.raw_rc, &other.raw_rc)
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
    fn ge(&self, other: &Arc<T, A>) -> bool {
        RawRc::ge(&self.raw_rc, &other.raw_rc)
    }
}
#[stable(feature = "rust1", since = "1.0.0")]
impl<T: ?Sized + Ord, A: Allocator> Ord for Arc<T, A> {
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
    fn cmp(&self, other: &Arc<T, A>) -> Ordering {
        RawRc::cmp(&self.raw_rc, &other.raw_rc)
    }
}
#[stable(feature = "rust1", since = "1.0.0")]
impl<T: ?Sized + Eq, A: Allocator> Eq for Arc<T, A> {}

#[stable(feature = "rust1", since = "1.0.0")]
impl<T: ?Sized + fmt::Display, A: Allocator> fmt::Display for Arc<T, A> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        <RawRc<T, A> as fmt::Display>::fmt(&self.raw_rc, f)
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<T: ?Sized + fmt::Debug, A: Allocator> fmt::Debug for Arc<T, A> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        <RawRc<T, A> as fmt::Debug>::fmt(&self.raw_rc, f)
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<T: ?Sized, A: Allocator> fmt::Pointer for Arc<T, A> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        <RawRc<T, A> as fmt::Pointer>::fmt(&self.raw_rc, f)
    }
}

#[cfg(not(no_global_oom_handling))]
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
        Self { raw_rc: RawRc::default() }
    }
}

#[cfg(not(no_global_oom_handling))]
const MAX_ALIGNMENT_FOR_STATIC_ALLOCATION: usize = 16;

/// Struct to hold the static `Arc` allocation used for empty `Arc<str/CStr/[T]>` as
/// returned by `Default::default`.
///
/// Layout notes:
///
/// * `repr(align(16))` so we can use it for `[T]` with `align_of::<T>() <= 16`.
/// * `repr(C)` so we can arrange reference counts and value deterministically.
/// * `[u8; 1]` (to be initialized with 0) so it can be used for `Arc<CStr>`.
#[cfg(not(no_global_oom_handling))]
#[repr(C, align(16))]
struct StaticSliceArcAllocation {
    padding: MaybeUninit<
        [u8; size_of::<RefCounts>()
            .checked_next_multiple_of(MAX_ALIGNMENT_FOR_STATIC_ALLOCATION)
            .unwrap()
            - size_of::<RefCounts>()],
    >,
    ref_counts: RefCounts,
    value: [u8; 1],
}

// Verify the memory layout of the static allocation.
#[cfg(not(no_global_oom_handling))]
const _: () = {
    const fn usize_max(lhs: usize, rhs: usize) -> usize {
        if lhs < rhs { rhs } else { lhs }
    }

    // Check the alignment of the allocation.
    assert!(
        align_of::<StaticSliceArcAllocation>()
            == usize_max(align_of::<RefCounts>(), MAX_ALIGNMENT_FOR_STATIC_ALLOCATION)
    );

    // Check the offset of the value.
    assert!(
        mem::offset_of!(StaticSliceArcAllocation, value)
            == size_of::<RefCounts>()
                .checked_next_multiple_of(MAX_ALIGNMENT_FOR_STATIC_ALLOCATION)
                .unwrap()
    );

    // Check the offset of the `RefCounts` object.
    assert!(
        mem::offset_of!(StaticSliceArcAllocation, ref_counts) + size_of::<RefCounts>()
            == mem::offset_of!(StaticSliceArcAllocation, value)
    );
};

#[cfg(not(no_global_oom_handling))]
unsafe impl Sync for StaticSliceArcAllocation {}

#[cfg(not(no_global_oom_handling))]
static STATIC_SLICE_ARC_ALLOCATION: StaticSliceArcAllocation = StaticSliceArcAllocation {
    padding: MaybeUninit::uninit(),
    ref_counts: RefCounts { weak: UnsafeCell::new(1), strong: UnsafeCell::new(1) },
    value: [0],
};

#[cfg(not(no_global_oom_handling))]
#[stable(feature = "more_rc_default_impls", since = "1.80.0")]
impl Default for Arc<str> {
    /// Creates an empty str inside an Arc
    ///
    /// This may or may not share an allocation with other Arcs.
    #[inline]
    fn default() -> Self {
        let arc: Arc<[u8]> = Default::default();
        debug_assert!(core::str::from_utf8(&*arc).is_ok());
        let (ptr, alloc) = Arc::into_raw_with_allocator(arc);
        unsafe { Arc::from_raw_in(ptr as *mut str, alloc) }
    }
}

#[cfg(not(no_global_oom_handling))]
#[stable(feature = "more_rc_default_impls", since = "1.80.0")]
impl Default for Arc<core::ffi::CStr> {
    /// Creates an empty CStr inside an Arc
    ///
    /// This may or may not share an allocation with other Arcs.
    #[inline]
    fn default() -> Self {
        unsafe {
            let ptr = NonNull::from(core::ffi::CStr::from_bytes_with_nul_unchecked(
                &STATIC_SLICE_ARC_ALLOCATION.value,
            ));

            Self { raw_rc: RawRc::from_raw(ptr).clone::<RefCounter>() }
        }
    }
}

#[cfg(not(no_global_oom_handling))]
#[stable(feature = "more_rc_default_impls", since = "1.80.0")]
impl<T> Default for Arc<[T]> {
    /// Creates an empty `[T]` inside an Arc
    ///
    /// This may or may not share an allocation with other Arcs.
    #[inline]
    fn default() -> Self {
        if align_of::<T>() <= MAX_ALIGNMENT_FOR_STATIC_ALLOCATION {
            unsafe {
                let ptr = NonNull::slice_from_raw_parts(
                    NonNull::from(&STATIC_SLICE_ARC_ALLOCATION.value).cast(),
                    0,
                );

                return Self { raw_rc: RawRc::from_raw(ptr).clone::<RefCounter>() };
            }
        }

        // If T's alignment is too large for the static, make a new unique allocation.
        Self { raw_rc: RawRc::default() }
    }
}

#[cfg(not(no_global_oom_handling))]
#[stable(feature = "pin_default_impls", since = "1.91.0")]
impl<T> Default for Pin<Arc<T>>
where
    T: ?Sized,
    Arc<T>: Default,
{
    #[inline]
    fn default() -> Self {
        unsafe { Pin::new_unchecked(Arc::<T>::default()) }
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<T: ?Sized + Hash, A: Allocator> Hash for Arc<T, A> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        RawRc::hash(&self.raw_rc, state);
    }
}

#[cfg(not(no_global_oom_handling))]
#[stable(feature = "from_for_ptrs", since = "1.6.0")]
impl<T> From<T> for Arc<T> {
    /// Converts a `T` into an `Arc<T>`
    ///
    /// The conversion moves the value into a
    /// newly allocated `Arc`. It is equivalent to
    /// calling `Arc::new(t)`.
    ///
    /// # Example
    /// ```rust
    /// # use std::sync::Arc;
    /// let x = 5;
    /// let arc = Arc::new(5);
    ///
    /// assert_eq!(Arc::from(x), arc);
    /// ```
    fn from(t: T) -> Self {
        Self { raw_rc: RawRc::from(t) }
    }
}

#[cfg(not(no_global_oom_handling))]
#[stable(feature = "shared_from_array", since = "1.74.0")]
impl<T, const N: usize> From<[T; N]> for Arc<[T]> {
    /// Converts a [`[T; N]`](prim@array) into an `Arc<[T]>`.
    ///
    /// The conversion moves the array into a newly allocated `Arc`.
    ///
    /// # Example
    ///
    /// ```
    /// # use std::sync::Arc;
    /// let original: [i32; 3] = [1, 2, 3];
    /// let shared: Arc<[i32]> = Arc::from(original);
    /// assert_eq!(&[1, 2, 3], &shared[..]);
    /// ```
    #[inline]
    fn from(v: [T; N]) -> Arc<[T]> {
        Self { raw_rc: RawRc::from(v) }
    }
}

#[cfg(not(no_global_oom_handling))]
#[stable(feature = "shared_from_slice", since = "1.21.0")]
impl<T: Clone> From<&[T]> for Arc<[T]> {
    /// Allocates a reference-counted slice and fills it by cloning `v`'s items.
    ///
    /// # Example
    ///
    /// ```
    /// # use std::sync::Arc;
    /// let original: &[i32] = &[1, 2, 3];
    /// let shared: Arc<[i32]> = Arc::from(original);
    /// assert_eq!(&[1, 2, 3], &shared[..]);
    /// ```
    #[inline]
    fn from(v: &[T]) -> Arc<[T]> {
        Self { raw_rc: RawRc::from(v) }
    }
}

#[cfg(not(no_global_oom_handling))]
#[stable(feature = "shared_from_mut_slice", since = "1.84.0")]
impl<T: Clone> From<&mut [T]> for Arc<[T]> {
    /// Allocates a reference-counted slice and fills it by cloning `v`'s items.
    ///
    /// # Example
    ///
    /// ```
    /// # use std::sync::Arc;
    /// let mut original = [1, 2, 3];
    /// let original: &mut [i32] = &mut original;
    /// let shared: Arc<[i32]> = Arc::from(original);
    /// assert_eq!(&[1, 2, 3], &shared[..]);
    /// ```
    #[inline]
    fn from(v: &mut [T]) -> Arc<[T]> {
        Self { raw_rc: RawRc::from(v) }
    }
}

#[cfg(not(no_global_oom_handling))]
#[stable(feature = "shared_from_slice", since = "1.21.0")]
impl From<&str> for Arc<str> {
    /// Allocates a reference-counted `str` and copies `v` into it.
    ///
    /// # Example
    ///
    /// ```
    /// # use std::sync::Arc;
    /// let shared: Arc<str> = Arc::from("eggplant");
    /// assert_eq!("eggplant", &shared[..]);
    /// ```
    #[inline]
    fn from(v: &str) -> Arc<str> {
        Self { raw_rc: RawRc::from(v) }
    }
}

#[cfg(not(no_global_oom_handling))]
#[stable(feature = "shared_from_mut_slice", since = "1.84.0")]
impl From<&mut str> for Arc<str> {
    /// Allocates a reference-counted `str` and copies `v` into it.
    ///
    /// # Example
    ///
    /// ```
    /// # use std::sync::Arc;
    /// let mut original = String::from("eggplant");
    /// let original: &mut str = &mut original;
    /// let shared: Arc<str> = Arc::from(original);
    /// assert_eq!("eggplant", &shared[..]);
    /// ```
    #[inline]
    fn from(v: &mut str) -> Arc<str> {
        Self { raw_rc: RawRc::from(v) }
    }
}

#[cfg(not(no_global_oom_handling))]
#[stable(feature = "shared_from_slice", since = "1.21.0")]
impl From<String> for Arc<str> {
    /// Allocates a reference-counted `str` and copies `v` into it.
    ///
    /// # Example
    ///
    /// ```
    /// # use std::sync::Arc;
    /// let unique: String = "eggplant".to_owned();
    /// let shared: Arc<str> = Arc::from(unique);
    /// assert_eq!("eggplant", &shared[..]);
    /// ```
    #[inline]
    fn from(v: String) -> Arc<str> {
        Self { raw_rc: RawRc::from(v) }
    }
}

#[cfg(not(no_global_oom_handling))]
#[stable(feature = "shared_from_slice", since = "1.21.0")]
impl<T: ?Sized, A: Allocator> From<Box<T, A>> for Arc<T, A> {
    /// Move a boxed object to a new, reference-counted allocation.
    ///
    /// # Example
    ///
    /// ```
    /// # use std::sync::Arc;
    /// let unique: Box<str> = Box::from("eggplant");
    /// let shared: Arc<str> = Arc::from(unique);
    /// assert_eq!("eggplant", &shared[..]);
    /// ```
    #[inline]
    fn from(v: Box<T, A>) -> Arc<T, A> {
        Self { raw_rc: RawRc::from(v) }
    }
}

#[cfg(not(no_global_oom_handling))]
#[stable(feature = "shared_from_slice", since = "1.21.0")]
impl<T, A: Allocator + Clone> From<Vec<T, A>> for Arc<[T], A> {
    /// Allocates a reference-counted slice and moves `v`'s items into it.
    ///
    /// # Example
    ///
    /// ```
    /// # use std::sync::Arc;
    /// let unique: Vec<i32> = vec![1, 2, 3];
    /// let shared: Arc<[i32]> = Arc::from(unique);
    /// assert_eq!(&[1, 2, 3], &shared[..]);
    /// ```
    #[inline]
    fn from(v: Vec<T, A>) -> Arc<[T], A> {
        Self { raw_rc: RawRc::from(v) }
    }
}

#[stable(feature = "shared_from_cow", since = "1.45.0")]
impl<'a, B> From<Cow<'a, B>> for Arc<B>
where
    B: ToOwned + ?Sized,
    Arc<B>: From<&'a B> + From<B::Owned>,
{
    /// Creates an atomically reference-counted pointer from a clone-on-write
    /// pointer by copying its content.
    ///
    /// # Example
    ///
    /// ```rust
    /// # use std::sync::Arc;
    /// # use std::borrow::Cow;
    /// let cow: Cow<'_, str> = Cow::Borrowed("eggplant");
    /// let shared: Arc<str> = Arc::from(cow);
    /// assert_eq!("eggplant", &shared[..]);
    /// ```
    #[inline]
    fn from(cow: Cow<'a, B>) -> Arc<B> {
        match cow {
            Cow::Borrowed(s) => Arc::from(s),
            Cow::Owned(s) => Arc::from(s),
        }
    }
}

#[stable(feature = "shared_from_str", since = "1.62.0")]
impl From<Arc<str>> for Arc<[u8]> {
    /// Converts an atomically reference-counted string slice into a byte slice.
    ///
    /// # Example
    ///
    /// ```
    /// # use std::sync::Arc;
    /// let string: Arc<str> = Arc::from("eggplant");
    /// let bytes: Arc<[u8]> = Arc::from(string);
    /// assert_eq!("eggplant".as_bytes(), bytes.as_ref());
    /// ```
    #[inline]
    fn from(rc: Arc<str>) -> Self {
        Self { raw_rc: RawRc::from(Arc::into_raw_rc(rc)) }
    }
}

#[stable(feature = "boxed_slice_try_from", since = "1.43.0")]
impl<T, A: Allocator, const N: usize> TryFrom<Arc<[T], A>> for Arc<[T; N], A> {
    type Error = Arc<[T], A>;

    fn try_from(boxed_slice: Arc<[T], A>) -> Result<Self, Self::Error> {
        match RawRc::try_from(Arc::into_raw_rc(boxed_slice)) {
            Ok(raw_rc) => Ok(Self { raw_rc }),
            Err(raw_rc) => Err(Arc { raw_rc }),
        }
    }
}

#[cfg(not(no_global_oom_handling))]
#[stable(feature = "shared_from_iter", since = "1.37.0")]
impl<T> FromIterator<T> for Arc<[T]> {
    /// Takes each element in the `Iterator` and collects it into an `Arc<[T]>`.
    ///
    /// # Performance characteristics
    ///
    /// ## The general case
    ///
    /// In the general case, collecting into `Arc<[T]>` is done by first
    /// collecting into a `Vec<T>`. That is, when writing the following:
    ///
    /// ```rust
    /// # use std::sync::Arc;
    /// let evens: Arc<[u8]> = (0..10).filter(|&x| x % 2 == 0).collect();
    /// # assert_eq!(&*evens, &[0, 2, 4, 6, 8]);
    /// ```
    ///
    /// this behaves as if we wrote:
    ///
    /// ```rust
    /// # use std::sync::Arc;
    /// let evens: Arc<[u8]> = (0..10).filter(|&x| x % 2 == 0)
    ///     .collect::<Vec<_>>() // The first set of allocations happens here.
    ///     .into(); // A second allocation for `Arc<[T]>` happens here.
    /// # assert_eq!(&*evens, &[0, 2, 4, 6, 8]);
    /// ```
    ///
    /// This will allocate as many times as needed for constructing the `Vec<T>`
    /// and then it will allocate once for turning the `Vec<T>` into the `Arc<[T]>`.
    ///
    /// ## Iterators of known length
    ///
    /// When your `Iterator` implements `TrustedLen` and is of an exact size,
    /// a single allocation will be made for the `Arc<[T]>`. For example:
    ///
    /// ```rust
    /// # use std::sync::Arc;
    /// let evens: Arc<[u8]> = (0..10).collect(); // Just a single allocation happens here.
    /// # assert_eq!(&*evens, &*(0..10).collect::<Vec<_>>());
    /// ```
    fn from_iter<I: IntoIterator<Item = T>>(iter: I) -> Self {
        Self { raw_rc: RawRc::from_iter(iter) }
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<T: ?Sized, A: Allocator> borrow::Borrow<T> for Arc<T, A> {
    fn borrow(&self) -> &T {
        self.raw_rc.as_ref()
    }
}

#[stable(since = "1.5.0", feature = "smart_ptr_as_ref")]
impl<T: ?Sized, A: Allocator> AsRef<T> for Arc<T, A> {
    fn as_ref(&self) -> &T {
        self.raw_rc.as_ref()
    }
}

#[stable(feature = "pin", since = "1.33.0")]
impl<T: ?Sized, A: Allocator> Unpin for Arc<T, A> {}

#[stable(feature = "arc_error", since = "1.52.0")]
impl<T: core::error::Error + ?Sized> core::error::Error for Arc<T> {
    #[allow(deprecated)]
    fn cause(&self) -> Option<&dyn core::error::Error> {
        RawRc::cause(&self.raw_rc)
    }

    fn source(&self) -> Option<&(dyn core::error::Error + 'static)> {
        RawRc::source(&self.raw_rc)
    }

    fn provide<'a>(&'a self, req: &mut core::error::Request<'a>) {
        RawRc::provide(&self.raw_rc, req)
    }
}

/// A uniquely owned [`Arc`].
///
/// This represents an `Arc` that is known to be uniquely owned -- that is, have exactly one strong
/// reference. Multiple weak pointers can be created, but attempts to upgrade those to strong
/// references will fail unless the `UniqueArc` they point to has been converted into a regular `Arc`.
///
/// Because it is uniquely owned, the contents of a `UniqueArc` can be freely mutated. A common
/// use case is to have an object be mutable during its initialization phase but then have it become
/// immutable and converted to a normal `Arc`.
///
/// This can be used as a flexible way to create cyclic data structures, as in the example below.
///
/// ```
/// #![feature(unique_rc_arc)]
/// use std::sync::{Arc, Weak, UniqueArc};
///
/// struct Gadget {
///     me: Weak<Gadget>,
/// }
///
/// fn create_gadget() -> Option<Arc<Gadget>> {
///     let mut rc = UniqueArc::new(Gadget {
///         me: Weak::new(),
///     });
///     rc.me = UniqueArc::downgrade(&rc);
///     Some(UniqueArc::into_arc(rc))
/// }
///
/// create_gadget().unwrap();
/// ```
///
/// An advantage of using `UniqueArc` over [`Arc::new_cyclic`] to build cyclic data structures is that
/// [`Arc::new_cyclic`]'s `data_fn` parameter cannot be async or return a [`Result`]. As shown in the
/// previous example, `UniqueArc` allows for more flexibility in the construction of cyclic data,
/// including fallible or async constructors.
#[unstable(feature = "unique_rc_arc", issue = "112566")]
pub struct UniqueArc<
    T: ?Sized,
    #[unstable(feature = "allocator_api", issue = "32838")] A: Allocator = Global,
> {
    raw_unique_rc: RawUniqueRc<T, A>,
}

#[unstable(feature = "unique_rc_arc", issue = "112566")]
unsafe impl<T: ?Sized + Sync + Send, A: Allocator + Send> Send for UniqueArc<T, A> {}

#[unstable(feature = "unique_rc_arc", issue = "112566")]
unsafe impl<T: ?Sized + Sync + Send, A: Allocator + Sync> Sync for UniqueArc<T, A> {}

#[unstable(feature = "unique_rc_arc", issue = "112566")]
// #[unstable(feature = "coerce_unsized", issue = "18598")]
impl<T: ?Sized + Unsize<U>, U: ?Sized, A: Allocator> CoerceUnsized<UniqueArc<U, A>>
    for UniqueArc<T, A>
{
}

//#[unstable(feature = "unique_rc_arc", issue = "112566")]
#[unstable(feature = "dispatch_from_dyn", issue = "none")]
impl<T: ?Sized + Unsize<U>, U: ?Sized> DispatchFromDyn<UniqueArc<U>> for UniqueArc<T> {}

#[unstable(feature = "unique_rc_arc", issue = "112566")]
impl<T: ?Sized + fmt::Display, A: Allocator> fmt::Display for UniqueArc<T, A> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        <RawUniqueRc<T, A> as fmt::Display>::fmt(&self.raw_unique_rc, f)
    }
}

#[unstable(feature = "unique_rc_arc", issue = "112566")]
impl<T: ?Sized + fmt::Debug, A: Allocator> fmt::Debug for UniqueArc<T, A> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        <RawUniqueRc<T, A> as fmt::Debug>::fmt(&self.raw_unique_rc, f)
    }
}

#[unstable(feature = "unique_rc_arc", issue = "112566")]
impl<T: ?Sized, A: Allocator> fmt::Pointer for UniqueArc<T, A> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        <RawUniqueRc<T, A> as fmt::Pointer>::fmt(&self.raw_unique_rc, f)
    }
}

#[unstable(feature = "unique_rc_arc", issue = "112566")]
impl<T: ?Sized, A: Allocator> borrow::Borrow<T> for UniqueArc<T, A> {
    fn borrow(&self) -> &T {
        self.raw_unique_rc.as_ref()
    }
}

#[unstable(feature = "unique_rc_arc", issue = "112566")]
impl<T: ?Sized, A: Allocator> borrow::BorrowMut<T> for UniqueArc<T, A> {
    fn borrow_mut(&mut self) -> &mut T {
        self.raw_unique_rc.as_mut()
    }
}

#[unstable(feature = "unique_rc_arc", issue = "112566")]
impl<T: ?Sized, A: Allocator> AsRef<T> for UniqueArc<T, A> {
    fn as_ref(&self) -> &T {
        self.raw_unique_rc.as_ref()
    }
}

#[unstable(feature = "unique_rc_arc", issue = "112566")]
impl<T: ?Sized, A: Allocator> AsMut<T> for UniqueArc<T, A> {
    fn as_mut(&mut self) -> &mut T {
        self.raw_unique_rc.as_mut()
    }
}

#[unstable(feature = "unique_rc_arc", issue = "112566")]
impl<T: ?Sized, A: Allocator> Unpin for UniqueArc<T, A> {}

#[unstable(feature = "unique_rc_arc", issue = "112566")]
impl<T: ?Sized + PartialEq, A: Allocator> PartialEq for UniqueArc<T, A> {
    /// Equality for two `UniqueArc`s.
    ///
    /// Two `UniqueArc`s are equal if their inner values are equal.
    ///
    /// # Examples
    ///
    /// ```
    /// #![feature(unique_rc_arc)]
    /// use std::sync::UniqueArc;
    ///
    /// let five = UniqueArc::new(5);
    ///
    /// assert!(five == UniqueArc::new(5));
    /// ```
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        RawUniqueRc::eq(&self.raw_unique_rc, &other.raw_unique_rc)
    }
}

#[unstable(feature = "unique_rc_arc", issue = "112566")]
impl<T: ?Sized + PartialOrd, A: Allocator> PartialOrd for UniqueArc<T, A> {
    /// Partial comparison for two `UniqueArc`s.
    ///
    /// The two are compared by calling `partial_cmp()` on their inner values.
    ///
    /// # Examples
    ///
    /// ```
    /// #![feature(unique_rc_arc)]
    /// use std::sync::UniqueArc;
    /// use std::cmp::Ordering;
    ///
    /// let five = UniqueArc::new(5);
    ///
    /// assert_eq!(Some(Ordering::Less), five.partial_cmp(&UniqueArc::new(6)));
    /// ```
    #[inline(always)]
    fn partial_cmp(&self, other: &UniqueArc<T, A>) -> Option<Ordering> {
        RawUniqueRc::partial_cmp(&self.raw_unique_rc, &other.raw_unique_rc)
    }

    /// Less-than comparison for two `UniqueArc`s.
    ///
    /// The two are compared by calling `<` on their inner values.
    ///
    /// # Examples
    ///
    /// ```
    /// #![feature(unique_rc_arc)]
    /// use std::sync::UniqueArc;
    ///
    /// let five = UniqueArc::new(5);
    ///
    /// assert!(five < UniqueArc::new(6));
    /// ```
    #[inline(always)]
    fn lt(&self, other: &UniqueArc<T, A>) -> bool {
        RawUniqueRc::lt(&self.raw_unique_rc, &other.raw_unique_rc)
    }

    /// 'Less than or equal to' comparison for two `UniqueArc`s.
    ///
    /// The two are compared by calling `<=` on their inner values.
    ///
    /// # Examples
    ///
    /// ```
    /// #![feature(unique_rc_arc)]
    /// use std::sync::UniqueArc;
    ///
    /// let five = UniqueArc::new(5);
    ///
    /// assert!(five <= UniqueArc::new(5));
    /// ```
    #[inline(always)]
    fn le(&self, other: &UniqueArc<T, A>) -> bool {
        RawUniqueRc::le(&self.raw_unique_rc, &other.raw_unique_rc)
    }

    /// Greater-than comparison for two `UniqueArc`s.
    ///
    /// The two are compared by calling `>` on their inner values.
    ///
    /// # Examples
    ///
    /// ```
    /// #![feature(unique_rc_arc)]
    /// use std::sync::UniqueArc;
    ///
    /// let five = UniqueArc::new(5);
    ///
    /// assert!(five > UniqueArc::new(4));
    /// ```
    #[inline(always)]
    fn gt(&self, other: &UniqueArc<T, A>) -> bool {
        RawUniqueRc::gt(&self.raw_unique_rc, &other.raw_unique_rc)
    }

    /// 'Greater than or equal to' comparison for two `UniqueArc`s.
    ///
    /// The two are compared by calling `>=` on their inner values.
    ///
    /// # Examples
    ///
    /// ```
    /// #![feature(unique_rc_arc)]
    /// use std::sync::UniqueArc;
    ///
    /// let five = UniqueArc::new(5);
    ///
    /// assert!(five >= UniqueArc::new(5));
    /// ```
    #[inline(always)]
    fn ge(&self, other: &UniqueArc<T, A>) -> bool {
        RawUniqueRc::ge(&self.raw_unique_rc, &other.raw_unique_rc)
    }
}

#[unstable(feature = "unique_rc_arc", issue = "112566")]
impl<T: ?Sized + Ord, A: Allocator> Ord for UniqueArc<T, A> {
    /// Comparison for two `UniqueArc`s.
    ///
    /// The two are compared by calling `cmp()` on their inner values.
    ///
    /// # Examples
    ///
    /// ```
    /// #![feature(unique_rc_arc)]
    /// use std::sync::UniqueArc;
    /// use std::cmp::Ordering;
    ///
    /// let five = UniqueArc::new(5);
    ///
    /// assert_eq!(Ordering::Less, five.cmp(&UniqueArc::new(6)));
    /// ```
    #[inline]
    fn cmp(&self, other: &UniqueArc<T, A>) -> Ordering {
        RawUniqueRc::cmp(&self.raw_unique_rc, &other.raw_unique_rc)
    }
}

#[unstable(feature = "unique_rc_arc", issue = "112566")]
impl<T: ?Sized + Eq, A: Allocator> Eq for UniqueArc<T, A> {}

#[unstable(feature = "unique_rc_arc", issue = "112566")]
impl<T: ?Sized + Hash, A: Allocator> Hash for UniqueArc<T, A> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        RawUniqueRc::hash(&self.raw_unique_rc, state);
    }
}

impl<T> UniqueArc<T, Global> {
    /// Creates a new `UniqueArc`.
    ///
    /// Weak references to this `UniqueArc` can be created with [`UniqueArc::downgrade`]. Upgrading
    /// these weak references will fail before the `UniqueArc` has been converted into an [`Arc`].
    /// After converting the `UniqueArc` into an [`Arc`], any weak references created beforehand will
    /// point to the new [`Arc`].
    #[cfg(not(no_global_oom_handling))]
    #[unstable(feature = "unique_rc_arc", issue = "112566")]
    #[must_use]
    pub fn new(value: T) -> Self {
        Self { raw_unique_rc: RawUniqueRc::new(value) }
    }

    /// Maps the value in a `UniqueArc`, reusing the allocation if possible.
    ///
    /// `f` is called on a reference to the value in the `UniqueArc`, and the result is returned,
    /// also in a `UniqueArc`.
    ///
    /// Note: this is an associated function, which means that you have
    /// to call it as `UniqueArc::map(u, f)` instead of `u.map(f)`. This
    /// is so that there is no conflict with a method on the inner type.
    ///
    /// # Examples
    ///
    /// ```
    /// #![feature(smart_pointer_try_map)]
    /// #![feature(unique_rc_arc)]
    ///
    /// use std::sync::UniqueArc;
    ///
    /// let r = UniqueArc::new(7);
    /// let new = UniqueArc::map(r, |i| i + 7);
    /// assert_eq!(*new, 14);
    /// ```
    #[cfg(not(no_global_oom_handling))]
    #[unstable(feature = "smart_pointer_try_map", issue = "144419")]
    pub fn map<U>(this: Self, f: impl FnOnce(T) -> U) -> UniqueArc<U> {
        let raw_unique_rc = Self::into_raw_unique_rc(this);

        UniqueArc { raw_unique_rc: unsafe { raw_unique_rc.map::<RefCounter, U>(f) } }
    }

    /// Attempts to map the value in a `UniqueArc`, reusing the allocation if possible.
    ///
    /// `f` is called on a reference to the value in the `UniqueArc`, and if the operation succeeds,
    /// the result is returned, also in a `UniqueArc`.
    ///
    /// Note: this is an associated function, which means that you have
    /// to call it as `UniqueArc::try_map(u, f)` instead of `u.try_map(f)`. This
    /// is so that there is no conflict with a method on the inner type.
    ///
    /// # Examples
    ///
    /// ```
    /// #![feature(smart_pointer_try_map)]
    /// #![feature(unique_rc_arc)]
    ///
    /// use std::sync::UniqueArc;
    ///
    /// let b = UniqueArc::new(7);
    /// let new = UniqueArc::try_map(b, u32::try_from).unwrap();
    /// assert_eq!(*new, 7);
    /// ```
    #[cfg(not(no_global_oom_handling))]
    #[unstable(feature = "smart_pointer_try_map", issue = "144419")]
    pub fn try_map<R>(
        this: Self,
        f: impl FnOnce(T) -> R,
    ) -> <R::Residual as Residual<UniqueArc<R::Output>>>::TryType
    where
        R: Try,
        R::Residual: Residual<UniqueArc<R::Output>>,
    {
        let raw_unique_rc = Self::into_raw_unique_rc(this);

        match unsafe { raw_unique_rc.try_map::<RefCounter, R>(f) } {
            ControlFlow::Continue(raw_unique_rc) => Try::from_output(UniqueArc { raw_unique_rc }),
            ControlFlow::Break(residual) => FromResidual::from_residual(residual),
        }
    }
}

impl<T, A: Allocator> UniqueArc<T, A> {
    /// Creates a new `UniqueArc` in the provided allocator.
    ///
    /// Weak references to this `UniqueArc` can be created with [`UniqueArc::downgrade`]. Upgrading
    /// these weak references will fail before the `UniqueArc` has been converted into an [`Arc`].
    /// After converting the `UniqueArc` into an [`Arc`], any weak references created beforehand will
    /// point to the new [`Arc`].
    #[cfg(not(no_global_oom_handling))]
    #[unstable(feature = "unique_rc_arc", issue = "112566")]
    #[must_use]
    // #[unstable(feature = "allocator_api", issue = "32838")]
    pub fn new_in(data: T, alloc: A) -> Self {
        Self { raw_unique_rc: RawUniqueRc::new_in(data, alloc) }
    }
}

impl<T: ?Sized, A: Allocator> UniqueArc<T, A> {
    #[cfg(not(no_global_oom_handling))]
    #[inline]
    fn into_raw_unique_rc(this: Self) -> RawUniqueRc<T, A> {
        let this = ManuallyDrop::new(this);

        unsafe { ptr::read(&this.raw_unique_rc) }
    }

    /// Converts the `UniqueArc` into a regular [`Arc`].
    ///
    /// This consumes the `UniqueArc` and returns a regular [`Arc`] that contains the `value` that
    /// is passed to `into_arc`.
    ///
    /// Any weak references created before this method is called can now be upgraded to strong
    /// references.
    #[unstable(feature = "unique_rc_arc", issue = "112566")]
    #[must_use]
    pub fn into_arc(this: Self) -> Arc<T, A> {
        let this = ManuallyDrop::new(this);
        let raw_rc = unsafe { ptr::read(&this.raw_unique_rc).into_rc::<RefCounter>() };

        Arc { raw_rc }
    }
}

impl<T: ?Sized, A: Allocator + Clone> UniqueArc<T, A> {
    /// Creates a new weak reference to the `UniqueArc`.
    ///
    /// Attempting to upgrade this weak reference will fail before the `UniqueArc` has been converted
    /// to a [`Arc`] using [`UniqueArc::into_arc`].
    #[unstable(feature = "unique_rc_arc", issue = "112566")]
    #[must_use]
    pub fn downgrade(this: &Self) -> Weak<T, A> {
        // SAFETY: The underlying implementation does not check whether the weak counter is locked.
        // It is safe because only `Arc::get_mut` locks the weak count, and as long as `UniqueArc`
        // exists, no `Arc` pointing to the same value can exist, so no locking should happen.
        let raw_weak = unsafe { this.raw_unique_rc.downgrade::<RefCounter>() };

        Weak { raw_weak }
    }
}

#[unstable(feature = "unique_rc_arc", issue = "112566")]
impl<T: ?Sized, A: Allocator> Deref for UniqueArc<T, A> {
    type Target = T;

    fn deref(&self) -> &T {
        self.raw_unique_rc.as_ref()
    }
}

// #[unstable(feature = "unique_rc_arc", issue = "112566")]
#[unstable(feature = "pin_coerce_unsized_trait", issue = "150112")]
unsafe impl<T: ?Sized> PinCoerceUnsized for UniqueArc<T> {}

#[unstable(feature = "unique_rc_arc", issue = "112566")]
impl<T: ?Sized, A: Allocator> DerefMut for UniqueArc<T, A> {
    fn deref_mut(&mut self) -> &mut T {
        self.raw_unique_rc.as_mut()
    }
}

#[unstable(feature = "unique_rc_arc", issue = "112566")]
// #[unstable(feature = "deref_pure_trait", issue = "87121")]
unsafe impl<T: ?Sized, A: Allocator> DerefPure for UniqueArc<T, A> {}

#[unstable(feature = "unique_rc_arc", issue = "112566")]
unsafe impl<#[may_dangle] T: ?Sized, A: Allocator> Drop for UniqueArc<T, A> {
    fn drop(&mut self) {
        unsafe { self.raw_unique_rc.drop::<RefCounter>() };
    }
}

#[unstable(feature = "allocator_api", issue = "32838")]
unsafe impl<T: ?Sized + Allocator, A: Allocator> Allocator for Arc<T, A> {
    #[inline]
    fn allocate(&self, layout: Layout) -> Result<NonNull<[u8]>, AllocError> {
        (**self).allocate(layout)
    }

    #[inline]
    fn allocate_zeroed(&self, layout: Layout) -> Result<NonNull<[u8]>, AllocError> {
        (**self).allocate_zeroed(layout)
    }

    #[inline]
    unsafe fn deallocate(&self, ptr: NonNull<u8>, layout: Layout) {
        // SAFETY: the safety contract must be upheld by the caller
        unsafe { (**self).deallocate(ptr, layout) }
    }

    #[inline]
    unsafe fn grow(
        &self,
        ptr: NonNull<u8>,
        old_layout: Layout,
        new_layout: Layout,
    ) -> Result<NonNull<[u8]>, AllocError> {
        // SAFETY: the safety contract must be upheld by the caller
        unsafe { (**self).grow(ptr, old_layout, new_layout) }
    }

    #[inline]
    unsafe fn grow_zeroed(
        &self,
        ptr: NonNull<u8>,
        old_layout: Layout,
        new_layout: Layout,
    ) -> Result<NonNull<[u8]>, AllocError> {
        // SAFETY: the safety contract must be upheld by the caller
        unsafe { (**self).grow_zeroed(ptr, old_layout, new_layout) }
    }

    #[inline]
    unsafe fn shrink(
        &self,
        ptr: NonNull<u8>,
        old_layout: Layout,
        new_layout: Layout,
    ) -> Result<NonNull<[u8]>, AllocError> {
        // SAFETY: the safety contract must be upheld by the caller
        unsafe { (**self).shrink(ptr, old_layout, new_layout) }
    }
}
