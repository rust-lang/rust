use cfg_if::cfg_if;

use crate::cell::UnsafeCell;
use crate::fmt;
use crate::ops::Deref;
use crate::panic::{RefUnwindSafe, UnwindSafe};
use crate::sys::sync as sys;
use crate::thread::{ThreadId, current_id};

/// A re-entrant mutual exclusion lock
///
/// This lock will block *other* threads waiting for the lock to become
/// available. The thread which has already locked the mutex can lock it
/// multiple times without blocking, preventing a common source of deadlocks.
///
/// # Examples
///
/// Allow recursively calling a function needing synchronization from within
/// a callback (this is how [`StdoutLock`](crate::io::StdoutLock) is currently
/// implemented):
///
/// ```
/// #![feature(reentrant_lock)]
///
/// use std::cell::RefCell;
/// use std::sync::ReentrantLock;
///
/// pub struct Log {
///     data: RefCell<String>,
/// }
///
/// impl Log {
///     pub fn append(&self, msg: &str) {
///         self.data.borrow_mut().push_str(msg);
///     }
/// }
///
/// static LOG: ReentrantLock<Log> = ReentrantLock::new(Log { data: RefCell::new(String::new()) });
///
/// pub fn with_log<R>(f: impl FnOnce(&Log) -> R) -> R {
///     let log = LOG.lock();
///     f(&*log)
/// }
///
/// with_log(|log| {
///     log.append("Hello");
///     with_log(|log| log.append(" there!"));
/// });
/// ```
///
// # Implementation details
//
// The 'owner' field tracks which thread has locked the mutex.
//
// We use thread::current_id() as the thread identifier, which is just the
// current thread's ThreadId, so it's unique across the process lifetime.
//
// If `owner` is set to the identifier of the current thread,
// we assume the mutex is already locked and instead of locking it again,
// we increment `lock_count`.
//
// When unlocking, we decrement `lock_count`, and only unlock the mutex when
// it reaches zero.
//
// `lock_count` is protected by the mutex and only accessed by the thread that has
// locked the mutex, so needs no synchronization.
//
// `owner` can be checked by other threads that want to see if they already
// hold the lock, so needs to be atomic. If it compares equal, we're on the
// same thread that holds the mutex and memory access can use relaxed ordering
// since we're not dealing with multiple threads. If it's not equal,
// synchronization is left to the mutex, making relaxed memory ordering for
// the `owner` field fine in all cases.
//
// On systems without 64 bit atomics we also store the address of a TLS variable
// along the 64-bit TID. We then first check that address against the address
// of that variable on the current thread, and only if they compare equal do we
// compare the actual TIDs. Because we only ever read the TID on the same thread
// that it was written on (or a thread sharing the TLS block with that writer thread),
// we don't need to further synchronize the TID accesses, so they can be regular 64-bit
// non-atomic accesses.
#[unstable(feature = "reentrant_lock", issue = "121440")]
pub struct ReentrantLock<T: ?Sized> {
    mutex: sys::Mutex,
    owner: Tid,
    lock_count: UnsafeCell<u32>,
    data: T,
}

cfg_if!(
    if #[cfg(target_has_atomic = "64")] {
        use crate::sync::atomic::{Atomic, AtomicU64, Ordering::Relaxed};

        struct Tid(Atomic<u64>);

        impl Tid {
            const fn new() -> Self {
                Self(AtomicU64::new(0))
            }

            #[inline]
            fn contains(&self, owner: ThreadId) -> bool {
                owner.as_u64().get() == self.0.load(Relaxed)
            }

            #[inline]
            // This is just unsafe to match the API of the Tid type below.
            unsafe fn set(&self, tid: Option<ThreadId>) {
                let value = tid.map_or(0, |tid| tid.as_u64().get());
                self.0.store(value, Relaxed);
            }
        }
    } else {
        /// Returns the address of a TLS variable. This is guaranteed to
        /// be unique across all currently alive threads.
        fn tls_addr() -> usize {
            thread_local! { static X: u8 = const { 0u8 } };

            X.with(|p| <*const u8>::addr(p))
        }

        use crate::sync::atomic::{
            Atomic,
            AtomicUsize,
            Ordering,
        };

        struct Tid {
            // When a thread calls `set()`, this value gets updated to
            // the address of a thread local on that thread. This is
            // used as a first check in `contains()`; if the `tls_addr`
            // doesn't match the TLS address of the current thread, then
            // the ThreadId also can't match. Only if the TLS addresses do
            // match do we read out the actual TID.
            // Note also that we can use relaxed atomic operations here, because
            // we only ever read from the tid if `tls_addr` matches the current
            // TLS address. In that case, either the tid has been set by
            // the current thread, or by a thread that has terminated before
            // the current thread was created. In either case, no further
            // synchronization is needed (as per <https://github.com/rust-lang/miri/issues/3450>)
            tls_addr: Atomic<usize>,
            tid: UnsafeCell<u64>,
        }

        unsafe impl Send for Tid {}
        unsafe impl Sync for Tid {}

        impl Tid {
            const fn new() -> Self {
                Self { tls_addr: AtomicUsize::new(0), tid: UnsafeCell::new(0) }
            }

            #[inline]
            // NOTE: This assumes that `owner` is the ID of the current
            // thread, and may spuriously return `false` if that's not the case.
            fn contains(&self, owner: ThreadId) -> bool {
                // SAFETY: See the comments in the struct definition.
                self.tls_addr.load(Ordering::Relaxed) == tls_addr()
                    && unsafe { *self.tid.get() } == owner.as_u64().get()
            }

            #[inline]
            // This may only be called by one thread at a time, and can lead to
            // race conditions otherwise.
            unsafe fn set(&self, tid: Option<ThreadId>) {
                // It's important that we set `self.tls_addr` to 0 if the tid is
                // cleared. Otherwise, there might be race conditions between
                // `set()` and `get()`.
                let tls_addr = if tid.is_some() { tls_addr() } else { 0 };
                let value = tid.map_or(0, |tid| tid.as_u64().get());
                self.tls_addr.store(tls_addr, Ordering::Relaxed);
                unsafe { *self.tid.get() = value };
            }
        }
    }
);

#[unstable(feature = "reentrant_lock", issue = "121440")]
unsafe impl<T: Send + ?Sized> Send for ReentrantLock<T> {}
#[unstable(feature = "reentrant_lock", issue = "121440")]
unsafe impl<T: Send + ?Sized> Sync for ReentrantLock<T> {}

// Because of the `UnsafeCell`, these traits are not implemented automatically
#[unstable(feature = "reentrant_lock", issue = "121440")]
impl<T: UnwindSafe + ?Sized> UnwindSafe for ReentrantLock<T> {}
#[unstable(feature = "reentrant_lock", issue = "121440")]
impl<T: RefUnwindSafe + ?Sized> RefUnwindSafe for ReentrantLock<T> {}

/// An RAII implementation of a "scoped lock" of a re-entrant lock. When this
/// structure is dropped (falls out of scope), the lock will be unlocked.
///
/// The data protected by the mutex can be accessed through this guard via its
/// [`Deref`] implementation.
///
/// This structure is created by the [`lock`](ReentrantLock::lock) method on
/// [`ReentrantLock`].
///
/// # Mutability
///
/// Unlike [`MutexGuard`](super::MutexGuard), `ReentrantLockGuard` does not
/// implement [`DerefMut`](crate::ops::DerefMut), because implementation of
/// the trait would violate Rust’s reference aliasing rules. Use interior
/// mutability (usually [`RefCell`](crate::cell::RefCell)) in order to mutate
/// the guarded data.
#[must_use = "if unused the ReentrantLock will immediately unlock"]
#[unstable(feature = "reentrant_lock", issue = "121440")]
pub struct ReentrantLockGuard<'a, T: ?Sized + 'a> {
    lock: &'a ReentrantLock<T>,
}

#[unstable(feature = "reentrant_lock", issue = "121440")]
impl<T: ?Sized> !Send for ReentrantLockGuard<'_, T> {}

#[unstable(feature = "reentrant_lock", issue = "121440")]
unsafe impl<T: ?Sized + Sync> Sync for ReentrantLockGuard<'_, T> {}

#[unstable(feature = "reentrant_lock", issue = "121440")]
impl<T> ReentrantLock<T> {
    /// Creates a new re-entrant lock in an unlocked state ready for use.
    ///
    /// # Examples
    ///
    /// ```
    /// #![feature(reentrant_lock)]
    /// use std::sync::ReentrantLock;
    ///
    /// let lock = ReentrantLock::new(0);
    /// ```
    pub const fn new(t: T) -> ReentrantLock<T> {
        ReentrantLock {
            mutex: sys::Mutex::new(),
            owner: Tid::new(),
            lock_count: UnsafeCell::new(0),
            data: t,
        }
    }

    /// Consumes this lock, returning the underlying data.
    ///
    /// # Examples
    ///
    /// ```
    /// #![feature(reentrant_lock)]
    ///
    /// use std::sync::ReentrantLock;
    ///
    /// let lock = ReentrantLock::new(0);
    /// assert_eq!(lock.into_inner(), 0);
    /// ```
    pub fn into_inner(self) -> T {
        self.data
    }
}

#[unstable(feature = "reentrant_lock", issue = "121440")]
impl<T: ?Sized> ReentrantLock<T> {
    /// Acquires the lock, blocking the current thread until it is able to do
    /// so.
    ///
    /// This function will block the caller until it is available to acquire
    /// the lock. Upon returning, the thread is the only thread with the lock
    /// held. When the thread calling this method already holds the lock, the
    /// call succeeds without blocking.
    ///
    /// # Examples
    ///
    /// ```
    /// #![feature(reentrant_lock)]
    /// use std::cell::Cell;
    /// use std::sync::{Arc, ReentrantLock};
    /// use std::thread;
    ///
    /// let lock = Arc::new(ReentrantLock::new(Cell::new(0)));
    /// let c_lock = Arc::clone(&lock);
    ///
    /// thread::spawn(move || {
    ///     c_lock.lock().set(10);
    /// }).join().expect("thread::spawn failed");
    /// assert_eq!(lock.lock().get(), 10);
    /// ```
    pub fn lock(&self) -> ReentrantLockGuard<'_, T> {
        let this_thread = current_id();
        // Safety: We only touch lock_count when we own the inner mutex.
        // Additionally, we only call `self.owner.set()` while holding
        // the inner mutex, so no two threads can call it concurrently.
        unsafe {
            if self.owner.contains(this_thread) {
                self.increment_lock_count().expect("lock count overflow in reentrant mutex");
            } else {
                self.mutex.lock();
                self.owner.set(Some(this_thread));
                debug_assert_eq!(*self.lock_count.get(), 0);
                *self.lock_count.get() = 1;
            }
        }
        ReentrantLockGuard { lock: self }
    }

    /// Returns a mutable reference to the underlying data.
    ///
    /// Since this call borrows the `ReentrantLock` mutably, no actual locking
    /// needs to take place -- the mutable borrow statically guarantees no locks
    /// exist.
    ///
    /// # Examples
    ///
    /// ```
    /// #![feature(reentrant_lock)]
    /// use std::sync::ReentrantLock;
    ///
    /// let mut lock = ReentrantLock::new(0);
    /// *lock.get_mut() = 10;
    /// assert_eq!(*lock.lock(), 10);
    /// ```
    pub fn get_mut(&mut self) -> &mut T {
        &mut self.data
    }

    /// Attempts to acquire this lock.
    ///
    /// If the lock could not be acquired at this time, then `None` is returned.
    /// Otherwise, an RAII guard is returned.
    ///
    /// This function does not block.
    // FIXME maybe make it a public part of the API?
    #[unstable(issue = "none", feature = "std_internals")]
    #[doc(hidden)]
    pub fn try_lock(&self) -> Option<ReentrantLockGuard<'_, T>> {
        let this_thread = current_id();
        // Safety: We only touch lock_count when we own the inner mutex.
        // Additionally, we only call `self.owner.set()` while holding
        // the inner mutex, so no two threads can call it concurrently.
        unsafe {
            if self.owner.contains(this_thread) {
                self.increment_lock_count()?;
                Some(ReentrantLockGuard { lock: self })
            } else if self.mutex.try_lock() {
                self.owner.set(Some(this_thread));
                debug_assert_eq!(*self.lock_count.get(), 0);
                *self.lock_count.get() = 1;
                Some(ReentrantLockGuard { lock: self })
            } else {
                None
            }
        }
    }

    unsafe fn increment_lock_count(&self) -> Option<()> {
        unsafe {
            *self.lock_count.get() = (*self.lock_count.get()).checked_add(1)?;
        }
        Some(())
    }
}

#[unstable(feature = "reentrant_lock", issue = "121440")]
impl<T: fmt::Debug + ?Sized> fmt::Debug for ReentrantLock<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut d = f.debug_struct("ReentrantLock");
        match self.try_lock() {
            Some(v) => d.field("data", &&*v),
            None => d.field("data", &format_args!("<locked>")),
        };
        d.finish_non_exhaustive()
    }
}

#[unstable(feature = "reentrant_lock", issue = "121440")]
impl<T: Default> Default for ReentrantLock<T> {
    fn default() -> Self {
        Self::new(T::default())
    }
}

#[unstable(feature = "reentrant_lock", issue = "121440")]
impl<T> From<T> for ReentrantLock<T> {
    fn from(t: T) -> Self {
        Self::new(t)
    }
}

#[unstable(feature = "reentrant_lock", issue = "121440")]
impl<T: ?Sized> Deref for ReentrantLockGuard<'_, T> {
    type Target = T;

    fn deref(&self) -> &T {
        &self.lock.data
    }
}

#[unstable(feature = "reentrant_lock", issue = "121440")]
impl<T: fmt::Debug + ?Sized> fmt::Debug for ReentrantLockGuard<'_, T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        (**self).fmt(f)
    }
}

#[unstable(feature = "reentrant_lock", issue = "121440")]
impl<T: fmt::Display + ?Sized> fmt::Display for ReentrantLockGuard<'_, T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        (**self).fmt(f)
    }
}

#[unstable(feature = "reentrant_lock", issue = "121440")]
impl<T: ?Sized> Drop for ReentrantLockGuard<'_, T> {
    #[inline]
    fn drop(&mut self) {
        // Safety: We own the lock.
        unsafe {
            *self.lock.lock_count.get() -= 1;
            if *self.lock.lock_count.get() == 0 {
                self.lock.owner.set(None);
                self.lock.mutex.unlock();
            }
        }
    }
}
