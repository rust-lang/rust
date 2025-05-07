//! Synchronization objects that employ poisoning.
//!
//! # Poisoning
//!
//! All synchronization objects in this module implement a strategy called "poisoning"
//! where if a thread panics while holding the exclusive access granted by the primitive,
//! the state of the primitive is set to "poisoned".
//! This information is then propagated to all other threads
//! to signify that the data protected by this primitive is likely tainted
//! (some invariant is not being upheld).
//!
//! The specifics of how this "poisoned" state affects other threads
//! depend on the primitive. See [#Overview] bellow.
//!
//! For the alternative implementations that do not employ poisoning,
//! see `std::sys::nonpoisoning`.
//!
//! # Overview
//!
//! Below is a list of synchronization objects provided by this module
//! with a high-level overview for each object and a description
//! of how it employs "poisoning".
//!
//! - [`Condvar`]: Condition Variable, providing the ability to block
//!   a thread while waiting for an event to occur.
//!
//!   Condition variables are typically associated with
//!   a boolean predicate (a condition) and a mutex.
//!   This implementation is associated with [`poison::Mutex`](Mutex),
//!   which employs poisoning.
//!   For this reason, [`Condvar::wait()`] will return a [`LockResult`],
//!   just like [`poison::Mutex::lock()`](Mutex::lock) does.
//!
//! - [`Mutex`]: Mutual Exclusion mechanism, which ensures that at
//!   most one thread at a time is able to access some data.
//!
//!   [`Mutex::lock()`] returns a [`LockResult`],
//!   providing a way to deal with the poisoned state.
//!   See [`Mutex`'s documentation](Mutex#poisoning) for more.
//!
//! - [`Once`]: A thread-safe way to run a piece of code only once.
//!   Mostly useful for implementing one-time global initialization.
//!
//!   [`Once`] is poisoned if the piece of code passed to
//!   [`Once::call_once()`] or [`Once::call_once_force()`] panics.
//!   When in poisoned state, subsequent calls to [`Once::call_once()`] will panic too.
//!   [`Once::call_once_force()`] can be used to clear the poisoned state.
//!
//! - [`RwLock`]: Provides a mutual exclusion mechanism which allows
//!   multiple readers at the same time, while allowing only one
//!   writer at a time. In some cases, this can be more efficient than
//!   a mutex.
//!
//!   This implementation, like [`Mutex`], will become poisoned on a panic.
//!   Note, however, that an `RwLock` may only be poisoned if a panic occurs
//!   while it is locked exclusively (write mode). If a panic occurs in any reader,
//!   then the lock will not be poisoned.

// FIXME(sync_nonpoison) add links to sync::nonpoison to the doc comment above.

#[stable(feature = "rust1", since = "1.0.0")]
pub use self::condvar::{Condvar, WaitTimeoutResult};
#[unstable(feature = "mapped_lock_guards", issue = "117108")]
pub use self::mutex::MappedMutexGuard;
#[stable(feature = "rust1", since = "1.0.0")]
pub use self::mutex::{Mutex, MutexGuard};
#[stable(feature = "rust1", since = "1.0.0")]
#[expect(deprecated)]
pub use self::once::ONCE_INIT;
#[stable(feature = "rust1", since = "1.0.0")]
pub use self::once::{Once, OnceState};
#[unstable(feature = "mapped_lock_guards", issue = "117108")]
pub use self::rwlock::{MappedRwLockReadGuard, MappedRwLockWriteGuard};
#[stable(feature = "rust1", since = "1.0.0")]
pub use self::rwlock::{RwLock, RwLockReadGuard, RwLockWriteGuard};
use crate::error::Error;
use crate::fmt;
#[cfg(panic = "unwind")]
use crate::sync::atomic::{Atomic, AtomicBool, Ordering};
#[cfg(panic = "unwind")]
use crate::thread;

mod condvar;
#[stable(feature = "rust1", since = "1.0.0")]
mod mutex;
pub(crate) mod once;
mod rwlock;

pub(crate) struct Flag {
    #[cfg(panic = "unwind")]
    failed: Atomic<bool>,
}

// Note that the Ordering uses to access the `failed` field of `Flag` below is
// always `Relaxed`, and that's because this isn't actually protecting any data,
// it's just a flag whether we've panicked or not.
//
// The actual location that this matters is when a mutex is **locked** which is
// where we have external synchronization ensuring that we see memory
// reads/writes to this flag.
//
// As a result, if it matters, we should see the correct value for `failed` in
// all cases.

impl Flag {
    #[inline]
    pub const fn new() -> Flag {
        Flag {
            #[cfg(panic = "unwind")]
            failed: AtomicBool::new(false),
        }
    }

    /// Checks the flag for an unguarded borrow, where we only care about existing poison.
    #[inline]
    pub fn borrow(&self) -> LockResult<()> {
        if self.get() { Err(PoisonError::new(())) } else { Ok(()) }
    }

    /// Checks the flag for a guarded borrow, where we may also set poison when `done`.
    #[inline]
    pub fn guard(&self) -> LockResult<Guard> {
        let ret = Guard {
            #[cfg(panic = "unwind")]
            panicking: thread::panicking(),
        };
        if self.get() { Err(PoisonError::new(ret)) } else { Ok(ret) }
    }

    #[inline]
    #[cfg(panic = "unwind")]
    pub fn done(&self, guard: &Guard) {
        if !guard.panicking && thread::panicking() {
            self.failed.store(true, Ordering::Relaxed);
        }
    }

    #[inline]
    #[cfg(not(panic = "unwind"))]
    pub fn done(&self, _guard: &Guard) {}

    #[inline]
    #[cfg(panic = "unwind")]
    pub fn get(&self) -> bool {
        self.failed.load(Ordering::Relaxed)
    }

    #[inline(always)]
    #[cfg(not(panic = "unwind"))]
    pub fn get(&self) -> bool {
        false
    }

    #[inline]
    pub fn clear(&self) {
        #[cfg(panic = "unwind")]
        self.failed.store(false, Ordering::Relaxed)
    }
}

#[derive(Clone)]
pub(crate) struct Guard {
    #[cfg(panic = "unwind")]
    panicking: bool,
}

/// A type of error which can be returned whenever a lock is acquired.
///
/// Both [`Mutex`]es and [`RwLock`]s are poisoned whenever a thread fails while the lock
/// is held. The precise semantics for when a lock is poisoned is documented on
/// each lock. For a lock in the poisoned state, unless the state is cleared manually,
/// all future acquisitions will return this error.
///
/// # Examples
///
/// ```
/// use std::sync::{Arc, Mutex};
/// use std::thread;
///
/// let mutex = Arc::new(Mutex::new(1));
///
/// // poison the mutex
/// let c_mutex = Arc::clone(&mutex);
/// let _ = thread::spawn(move || {
///     let mut data = c_mutex.lock().unwrap();
///     *data = 2;
///     panic!();
/// }).join();
///
/// match mutex.lock() {
///     Ok(_) => unreachable!(),
///     Err(p_err) => {
///         let data = p_err.get_ref();
///         println!("recovered: {data}");
///     }
/// };
/// ```
/// [`Mutex`]: crate::sync::Mutex
/// [`RwLock`]: crate::sync::RwLock
#[stable(feature = "rust1", since = "1.0.0")]
pub struct PoisonError<T> {
    data: T,
    #[cfg(not(panic = "unwind"))]
    _never: !,
}

/// An enumeration of possible errors associated with a [`TryLockResult`] which
/// can occur while trying to acquire a lock, from the [`try_lock`] method on a
/// [`Mutex`] or the [`try_read`] and [`try_write`] methods on an [`RwLock`].
///
/// [`try_lock`]: crate::sync::Mutex::try_lock
/// [`try_read`]: crate::sync::RwLock::try_read
/// [`try_write`]: crate::sync::RwLock::try_write
/// [`Mutex`]: crate::sync::Mutex
/// [`RwLock`]: crate::sync::RwLock
#[stable(feature = "rust1", since = "1.0.0")]
pub enum TryLockError<T> {
    /// The lock could not be acquired because another thread failed while holding
    /// the lock.
    #[stable(feature = "rust1", since = "1.0.0")]
    Poisoned(#[stable(feature = "rust1", since = "1.0.0")] PoisonError<T>),
    /// The lock could not be acquired at this time because the operation would
    /// otherwise block.
    #[stable(feature = "rust1", since = "1.0.0")]
    WouldBlock,
}

/// A type alias for the result of a lock method which can be poisoned.
///
/// The [`Ok`] variant of this result indicates that the primitive was not
/// poisoned, and the operation result is contained within. The [`Err`] variant indicates
/// that the primitive was poisoned. Note that the [`Err`] variant *also* carries
/// an associated value assigned by the lock method, and it can be acquired through the
/// [`into_inner`] method. The semantics of the associated value depends on the corresponding
/// lock method.
///
/// [`into_inner`]: PoisonError::into_inner
#[stable(feature = "rust1", since = "1.0.0")]
pub type LockResult<T> = Result<T, PoisonError<T>>;

/// A type alias for the result of a nonblocking locking method.
///
/// For more information, see [`LockResult`]. A `TryLockResult` doesn't
/// necessarily hold the associated guard in the [`Err`] type as the lock might not
/// have been acquired for other reasons.
#[stable(feature = "rust1", since = "1.0.0")]
pub type TryLockResult<Guard> = Result<Guard, TryLockError<Guard>>;

#[stable(feature = "rust1", since = "1.0.0")]
impl<T> fmt::Debug for PoisonError<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("PoisonError").finish_non_exhaustive()
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<T> fmt::Display for PoisonError<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        "poisoned lock: another task failed inside".fmt(f)
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<T> Error for PoisonError<T> {
    #[allow(deprecated)]
    fn description(&self) -> &str {
        "poisoned lock: another task failed inside"
    }
}

impl<T> PoisonError<T> {
    /// Creates a `PoisonError`.
    ///
    /// This is generally created by methods like [`Mutex::lock`](crate::sync::Mutex::lock)
    /// or [`RwLock::read`](crate::sync::RwLock::read).
    ///
    /// This method may panic if std was built with `panic="abort"`.
    #[cfg(panic = "unwind")]
    #[stable(feature = "sync_poison", since = "1.2.0")]
    pub fn new(data: T) -> PoisonError<T> {
        PoisonError { data }
    }

    /// Creates a `PoisonError`.
    ///
    /// This is generally created by methods like [`Mutex::lock`](crate::sync::Mutex::lock)
    /// or [`RwLock::read`](crate::sync::RwLock::read).
    ///
    /// This method may panic if std was built with `panic="abort"`.
    #[cfg(not(panic = "unwind"))]
    #[stable(feature = "sync_poison", since = "1.2.0")]
    #[track_caller]
    pub fn new(_data: T) -> PoisonError<T> {
        panic!("PoisonError created in a libstd built with panic=\"abort\"")
    }

    /// Consumes this error indicating that a lock is poisoned, returning the
    /// associated data.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::collections::HashSet;
    /// use std::sync::{Arc, Mutex};
    /// use std::thread;
    ///
    /// let mutex = Arc::new(Mutex::new(HashSet::new()));
    ///
    /// // poison the mutex
    /// let c_mutex = Arc::clone(&mutex);
    /// let _ = thread::spawn(move || {
    ///     let mut data = c_mutex.lock().unwrap();
    ///     data.insert(10);
    ///     panic!();
    /// }).join();
    ///
    /// let p_err = mutex.lock().unwrap_err();
    /// let data = p_err.into_inner();
    /// println!("recovered {} items", data.len());
    /// ```
    #[stable(feature = "sync_poison", since = "1.2.0")]
    pub fn into_inner(self) -> T {
        self.data
    }

    /// Reaches into this error indicating that a lock is poisoned, returning a
    /// reference to the associated data.
    #[stable(feature = "sync_poison", since = "1.2.0")]
    pub fn get_ref(&self) -> &T {
        &self.data
    }

    /// Reaches into this error indicating that a lock is poisoned, returning a
    /// mutable reference to the associated data.
    #[stable(feature = "sync_poison", since = "1.2.0")]
    pub fn get_mut(&mut self) -> &mut T {
        &mut self.data
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<T> From<PoisonError<T>> for TryLockError<T> {
    fn from(err: PoisonError<T>) -> TryLockError<T> {
        TryLockError::Poisoned(err)
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<T> fmt::Debug for TryLockError<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match *self {
            #[cfg(panic = "unwind")]
            TryLockError::Poisoned(..) => "Poisoned(..)".fmt(f),
            #[cfg(not(panic = "unwind"))]
            TryLockError::Poisoned(ref p) => match p._never {},
            TryLockError::WouldBlock => "WouldBlock".fmt(f),
        }
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<T> fmt::Display for TryLockError<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match *self {
            #[cfg(panic = "unwind")]
            TryLockError::Poisoned(..) => "poisoned lock: another task failed inside",
            #[cfg(not(panic = "unwind"))]
            TryLockError::Poisoned(ref p) => match p._never {},
            TryLockError::WouldBlock => "try_lock failed because the operation would block",
        }
        .fmt(f)
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<T> Error for TryLockError<T> {
    #[allow(deprecated, deprecated_in_future)]
    fn description(&self) -> &str {
        match *self {
            #[cfg(panic = "unwind")]
            TryLockError::Poisoned(ref p) => p.description(),
            #[cfg(not(panic = "unwind"))]
            TryLockError::Poisoned(ref p) => match p._never {},
            TryLockError::WouldBlock => "try_lock failed because the operation would block",
        }
    }

    #[allow(deprecated)]
    fn cause(&self) -> Option<&dyn Error> {
        match *self {
            #[cfg(panic = "unwind")]
            TryLockError::Poisoned(ref p) => Some(p),
            #[cfg(not(panic = "unwind"))]
            TryLockError::Poisoned(ref p) => match p._never {},
            _ => None,
        }
    }
}

pub(crate) fn map_result<T, U, F>(result: LockResult<T>, f: F) -> LockResult<U>
where
    F: FnOnce(T) -> U,
{
    match result {
        Ok(t) => Ok(f(t)),
        #[cfg(panic = "unwind")]
        Err(PoisonError { data }) => Err(PoisonError::new(f(data))),
    }
}
