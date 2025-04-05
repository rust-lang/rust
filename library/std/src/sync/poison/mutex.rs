use crate::cell::UnsafeCell;
use crate::fmt;
use crate::marker::PhantomData;
use crate::mem::{self, ManuallyDrop};
use crate::ops::{Deref, DerefMut};
use crate::ptr::NonNull;
use crate::sync::{LockResult, PoisonError, TryLockError, TryLockResult, poison};
use crate::sys::sync as sys;

/// A mutual exclusion primitive useful for protecting shared data
///
/// This mutex will block threads waiting for the lock to become available. The
/// mutex can be created via a [`new`] constructor. Each mutex has a type parameter
/// which represents the data that it is protecting. The data can only be accessed
/// through the RAII guards returned from [`lock`] and [`try_lock`], which
/// guarantees that the data is only ever accessed when the mutex is locked.
///
/// # Poisoning
///
/// The mutexes in this module implement a strategy called "poisoning" where a
/// mutex is considered poisoned whenever a thread panics while holding the
/// mutex. Once a mutex is poisoned, all other threads are unable to access the
/// data by default as it is likely tainted (some invariant is not being
/// upheld).
///
/// For a mutex, this means that the [`lock`] and [`try_lock`] methods return a
/// [`Result`] which indicates whether a mutex has been poisoned or not. Most
/// usage of a mutex will simply [`unwrap()`] these results, propagating panics
/// among threads to ensure that a possibly invalid invariant is not witnessed.
///
/// A poisoned mutex, however, does not prevent all access to the underlying
/// data. The [`PoisonError`] type has an [`into_inner`] method which will return
/// the guard that would have otherwise been returned on a successful lock. This
/// allows access to the data, despite the lock being poisoned.
///
/// [`new`]: Self::new
/// [`lock`]: Self::lock
/// [`try_lock`]: Self::try_lock
/// [`unwrap()`]: Result::unwrap
/// [`PoisonError`]: super::PoisonError
/// [`into_inner`]: super::PoisonError::into_inner
///
/// # Examples
///
/// ```
/// use std::sync::{Arc, Mutex};
/// use std::thread;
/// use std::sync::mpsc::channel;
///
/// const N: usize = 10;
///
/// // Spawn a few threads to increment a shared variable (non-atomically), and
/// // let the main thread know once all increments are done.
/// //
/// // Here we're using an Arc to share memory among threads, and the data inside
/// // the Arc is protected with a mutex.
/// let data = Arc::new(Mutex::new(0));
///
/// let (tx, rx) = channel();
/// for _ in 0..N {
///     let (data, tx) = (Arc::clone(&data), tx.clone());
///     thread::spawn(move || {
///         // The shared state can only be accessed once the lock is held.
///         // Our non-atomic increment is safe because we're the only thread
///         // which can access the shared state when the lock is held.
///         //
///         // We unwrap() the return value to assert that we are not expecting
///         // threads to ever fail while holding the lock.
///         let mut data = data.lock().unwrap();
///         *data += 1;
///         if *data == N {
///             tx.send(()).unwrap();
///         }
///         // the lock is unlocked here when `data` goes out of scope.
///     });
/// }
///
/// rx.recv().unwrap();
/// ```
///
/// To recover from a poisoned mutex:
///
/// ```
/// use std::sync::{Arc, Mutex};
/// use std::thread;
///
/// let lock = Arc::new(Mutex::new(0_u32));
/// let lock2 = Arc::clone(&lock);
///
/// let _ = thread::spawn(move || -> () {
///     // This thread will acquire the mutex first, unwrapping the result of
///     // `lock` because the lock has not been poisoned.
///     let _guard = lock2.lock().unwrap();
///
///     // This panic while holding the lock (`_guard` is in scope) will poison
///     // the mutex.
///     panic!();
/// }).join();
///
/// // The lock is poisoned by this point, but the returned result can be
/// // pattern matched on to return the underlying guard on both branches.
/// let mut guard = match lock.lock() {
///     Ok(guard) => guard,
///     Err(poisoned) => poisoned.into_inner(),
/// };
///
/// *guard += 1;
/// ```
///
/// To unlock a mutex guard sooner than the end of the enclosing scope,
/// either create an inner scope or drop the guard manually.
///
/// ```
/// use std::sync::{Arc, Mutex};
/// use std::thread;
///
/// const N: usize = 3;
///
/// let data_mutex = Arc::new(Mutex::new(vec![1, 2, 3, 4]));
/// let res_mutex = Arc::new(Mutex::new(0));
///
/// let mut threads = Vec::with_capacity(N);
/// (0..N).for_each(|_| {
///     let data_mutex_clone = Arc::clone(&data_mutex);
///     let res_mutex_clone = Arc::clone(&res_mutex);
///
///     threads.push(thread::spawn(move || {
///         // Here we use a block to limit the lifetime of the lock guard.
///         let result = {
///             let mut data = data_mutex_clone.lock().unwrap();
///             // This is the result of some important and long-ish work.
///             let result = data.iter().fold(0, |acc, x| acc + x * 2);
///             data.push(result);
///             result
///             // The mutex guard gets dropped here, together with any other values
///             // created in the critical section.
///         };
///         // The guard created here is a temporary dropped at the end of the statement, i.e.
///         // the lock would not remain being held even if the thread did some additional work.
///         *res_mutex_clone.lock().unwrap() += result;
///     }));
/// });
///
/// let mut data = data_mutex.lock().unwrap();
/// // This is the result of some important and long-ish work.
/// let result = data.iter().fold(0, |acc, x| acc + x * 2);
/// data.push(result);
/// // We drop the `data` explicitly because it's not necessary anymore and the
/// // thread still has work to do. This allows other threads to start working on
/// // the data immediately, without waiting for the rest of the unrelated work
/// // to be done here.
/// //
/// // It's even more important here than in the threads because we `.join` the
/// // threads after that. If we had not dropped the mutex guard, a thread could
/// // be waiting forever for it, causing a deadlock.
/// // As in the threads, a block could have been used instead of calling the
/// // `drop` function.
/// drop(data);
/// // Here the mutex guard is not assigned to a variable and so, even if the
/// // scope does not end after this line, the mutex is still released: there is
/// // no deadlock.
/// *res_mutex.lock().unwrap() += result;
///
/// threads.into_iter().for_each(|thread| {
///     thread
///         .join()
///         .expect("The thread creating or execution failed !")
/// });
///
/// assert_eq!(*res_mutex.lock().unwrap(), 800);
/// ```
///
#[stable(feature = "rust1", since = "1.0.0")]
#[cfg_attr(not(test), rustc_diagnostic_item = "Mutex")]
pub struct Mutex<T: ?Sized> {
    inner: sys::Mutex,
    poison: poison::Flag,
    data: UnsafeCell<T>,
}

/// `T` must be `Send` for a [`Mutex`] to be `Send` because it is possible to acquire
/// the owned `T` from the `Mutex` via [`into_inner`].
///
/// [`into_inner`]: Mutex::into_inner
#[stable(feature = "rust1", since = "1.0.0")]
unsafe impl<T: ?Sized + Send> Send for Mutex<T> {}

/// `T` must be `Send` for [`Mutex`] to be `Sync`.
/// This ensures that the protected data can be accessed safely from multiple threads
/// without causing data races or other unsafe behavior.
///
/// [`Mutex<T>`] provides mutable access to `T` to one thread at a time. However, it's essential
/// for `T` to be `Send` because it's not safe for non-`Send` structures to be accessed in
/// this manner. For instance, consider [`Rc`], a non-atomic reference counted smart pointer,
/// which is not `Send`. With `Rc`, we can have multiple copies pointing to the same heap
/// allocation with a non-atomic reference count. If we were to use `Mutex<Rc<_>>`, it would
/// only protect one instance of `Rc` from shared access, leaving other copies vulnerable
/// to potential data races.
///
/// Also note that it is not necessary for `T` to be `Sync` as `&T` is only made available
/// to one thread at a time if `T` is not `Sync`.
///
/// [`Rc`]: crate::rc::Rc
#[stable(feature = "rust1", since = "1.0.0")]
unsafe impl<T: ?Sized + Send> Sync for Mutex<T> {}

/// An RAII implementation of a "scoped lock" of a mutex. When this structure is
/// dropped (falls out of scope), the lock will be unlocked.
///
/// The data protected by the mutex can be accessed through this guard via its
/// [`Deref`] and [`DerefMut`] implementations.
///
/// This structure is created by the [`lock`] and [`try_lock`] methods on
/// [`Mutex`].
///
/// [`lock`]: Mutex::lock
/// [`try_lock`]: Mutex::try_lock
#[must_use = "if unused the Mutex will immediately unlock"]
#[must_not_suspend = "holding a MutexGuard across suspend \
                      points can cause deadlocks, delays, \
                      and cause Futures to not implement `Send`"]
#[stable(feature = "rust1", since = "1.0.0")]
#[clippy::has_significant_drop]
#[cfg_attr(not(test), rustc_diagnostic_item = "MutexGuard")]
pub struct MutexGuard<'a, T: ?Sized + 'a> {
    lock: &'a Mutex<T>,
    poison: poison::Guard,
}

/// A [`MutexGuard`] is not `Send` to maximize platform portablity.
///
/// On platforms that use POSIX threads (commonly referred to as pthreads) there is a requirement to
/// release mutex locks on the same thread they were acquired.
/// For this reason, [`MutexGuard`] must not implement `Send` to prevent it being dropped from
/// another thread.
#[stable(feature = "rust1", since = "1.0.0")]
impl<T: ?Sized> !Send for MutexGuard<'_, T> {}

/// `T` must be `Sync` for a [`MutexGuard<T>`] to be `Sync`
/// because it is possible to get a `&T` from `&MutexGuard` (via `Deref`).
#[stable(feature = "mutexguard", since = "1.19.0")]
unsafe impl<T: ?Sized + Sync> Sync for MutexGuard<'_, T> {}

/// An RAII mutex guard returned by `MutexGuard::map`, which can point to a
/// subfield of the protected data. When this structure is dropped (falls out
/// of scope), the lock will be unlocked.
///
/// The main difference between `MappedMutexGuard` and [`MutexGuard`] is that the
/// former cannot be used with [`Condvar`], since that
/// could introduce soundness issues if the locked object is modified by another
/// thread while the `Mutex` is unlocked.
///
/// The data protected by the mutex can be accessed through this guard via its
/// [`Deref`] and [`DerefMut`] implementations.
///
/// This structure is created by the [`map`] and [`try_map`] methods on
/// [`MutexGuard`].
///
/// [`map`]: MutexGuard::map
/// [`try_map`]: MutexGuard::try_map
/// [`Condvar`]: crate::sync::Condvar
#[must_use = "if unused the Mutex will immediately unlock"]
#[must_not_suspend = "holding a MappedMutexGuard across suspend \
                      points can cause deadlocks, delays, \
                      and cause Futures to not implement `Send`"]
#[unstable(feature = "mapped_lock_guards", issue = "117108")]
#[clippy::has_significant_drop]
pub struct MappedMutexGuard<'a, T: ?Sized + 'a> {
    // NB: we use a pointer instead of `&'a mut T` to avoid `noalias` violations, because a
    // `MappedMutexGuard` argument doesn't hold uniqueness for its whole scope, only until it drops.
    // `NonNull` is covariant over `T`, so we add a `PhantomData<&'a mut T>` field
    // below for the correct variance over `T` (invariance).
    data: NonNull<T>,
    inner: &'a sys::Mutex,
    poison_flag: &'a poison::Flag,
    poison: poison::Guard,
    _variance: PhantomData<&'a mut T>,
}

#[unstable(feature = "mapped_lock_guards", issue = "117108")]
impl<T: ?Sized> !Send for MappedMutexGuard<'_, T> {}
#[unstable(feature = "mapped_lock_guards", issue = "117108")]
unsafe impl<T: ?Sized + Sync> Sync for MappedMutexGuard<'_, T> {}

impl<T> Mutex<T> {
    /// Creates a new mutex in an unlocked state ready for use.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::sync::Mutex;
    ///
    /// let mutex = Mutex::new(0);
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    #[rustc_const_stable(feature = "const_locks", since = "1.63.0")]
    #[inline]
    pub const fn new(t: T) -> Mutex<T> {
        Mutex { inner: sys::Mutex::new(), poison: poison::Flag::new(), data: UnsafeCell::new(t) }
    }

    /// Returns the contained value by cloning it.
    ///
    /// # Errors
    ///
    /// If another user of this mutex panicked while holding the mutex, then
    /// this call will return an error instead.
    ///
    /// # Examples
    ///
    /// ```
    /// #![feature(lock_value_accessors)]
    ///
    /// use std::sync::Mutex;
    ///
    /// let mut mutex = Mutex::new(7);
    ///
    /// assert_eq!(mutex.get_cloned().unwrap(), 7);
    /// ```
    #[unstable(feature = "lock_value_accessors", issue = "133407")]
    pub fn get_cloned(&self) -> Result<T, PoisonError<()>>
    where
        T: Clone,
    {
        match self.lock() {
            Ok(guard) => Ok((*guard).clone()),
            Err(_) => Err(PoisonError::new(())),
        }
    }

    /// Sets the contained value.
    ///
    /// # Errors
    ///
    /// If another user of this mutex panicked while holding the mutex, then
    /// this call will return an error containing the provided `value` instead.
    ///
    /// # Examples
    ///
    /// ```
    /// #![feature(lock_value_accessors)]
    ///
    /// use std::sync::Mutex;
    ///
    /// let mut mutex = Mutex::new(7);
    ///
    /// assert_eq!(mutex.get_cloned().unwrap(), 7);
    /// mutex.set(11).unwrap();
    /// assert_eq!(mutex.get_cloned().unwrap(), 11);
    /// ```
    #[unstable(feature = "lock_value_accessors", issue = "133407")]
    pub fn set(&self, value: T) -> Result<(), PoisonError<T>> {
        if mem::needs_drop::<T>() {
            // If the contained value has non-trivial destructor, we
            // call that destructor after the lock being released.
            self.replace(value).map(drop)
        } else {
            match self.lock() {
                Ok(mut guard) => {
                    *guard = value;

                    Ok(())
                }
                Err(_) => Err(PoisonError::new(value)),
            }
        }
    }

    /// Replaces the contained value with `value`, and returns the old contained value.
    ///
    /// # Errors
    ///
    /// If another user of this mutex panicked while holding the mutex, then
    /// this call will return an error containing the provided `value` instead.
    ///
    /// # Examples
    ///
    /// ```
    /// #![feature(lock_value_accessors)]
    ///
    /// use std::sync::Mutex;
    ///
    /// let mut mutex = Mutex::new(7);
    ///
    /// assert_eq!(mutex.replace(11).unwrap(), 7);
    /// assert_eq!(mutex.get_cloned().unwrap(), 11);
    /// ```
    #[unstable(feature = "lock_value_accessors", issue = "133407")]
    pub fn replace(&self, value: T) -> LockResult<T> {
        match self.lock() {
            Ok(mut guard) => Ok(mem::replace(&mut *guard, value)),
            Err(_) => Err(PoisonError::new(value)),
        }
    }
}

impl<T: ?Sized> Mutex<T> {
    /// Acquires a mutex, blocking the current thread until it is able to do so.
    ///
    /// This function will block the local thread until it is available to acquire
    /// the mutex. Upon returning, the thread is the only thread with the lock
    /// held. An RAII guard is returned to allow scoped unlock of the lock. When
    /// the guard goes out of scope, the mutex will be unlocked.
    ///
    /// The exact behavior on locking a mutex in the thread which already holds
    /// the lock is left unspecified. However, this function will not return on
    /// the second call (it might panic or deadlock, for example).
    ///
    /// # Errors
    ///
    /// If another user of this mutex panicked while holding the mutex, then
    /// this call will return an error once the mutex is acquired. The acquired
    /// mutex guard will be contained in the returned error.
    ///
    /// # Panics
    ///
    /// This function might panic when called if the lock is already held by
    /// the current thread.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::sync::{Arc, Mutex};
    /// use std::thread;
    ///
    /// let mutex = Arc::new(Mutex::new(0));
    /// let c_mutex = Arc::clone(&mutex);
    ///
    /// thread::spawn(move || {
    ///     *c_mutex.lock().unwrap() = 10;
    /// }).join().expect("thread::spawn failed");
    /// assert_eq!(*mutex.lock().unwrap(), 10);
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn lock(&self) -> LockResult<MutexGuard<'_, T>> {
        unsafe {
            self.inner.lock();
            MutexGuard::new(self)
        }
    }

    /// Attempts to acquire this lock.
    ///
    /// If the lock could not be acquired at this time, then [`Err`] is returned.
    /// Otherwise, an RAII guard is returned. The lock will be unlocked when the
    /// guard is dropped.
    ///
    /// This function does not block.
    ///
    /// # Errors
    ///
    /// If another user of this mutex panicked while holding the mutex, then
    /// this call will return the [`Poisoned`] error if the mutex would
    /// otherwise be acquired. An acquired lock guard will be contained
    /// in the returned error.
    ///
    /// If the mutex could not be acquired because it is already locked, then
    /// this call will return the [`WouldBlock`] error.
    ///
    /// [`Poisoned`]: TryLockError::Poisoned
    /// [`WouldBlock`]: TryLockError::WouldBlock
    ///
    /// # Examples
    ///
    /// ```
    /// use std::sync::{Arc, Mutex};
    /// use std::thread;
    ///
    /// let mutex = Arc::new(Mutex::new(0));
    /// let c_mutex = Arc::clone(&mutex);
    ///
    /// thread::spawn(move || {
    ///     let mut lock = c_mutex.try_lock();
    ///     if let Ok(ref mut mutex) = lock {
    ///         **mutex = 10;
    ///     } else {
    ///         println!("try_lock failed");
    ///     }
    /// }).join().expect("thread::spawn failed");
    /// assert_eq!(*mutex.lock().unwrap(), 10);
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn try_lock(&self) -> TryLockResult<MutexGuard<'_, T>> {
        unsafe {
            if self.inner.try_lock() {
                Ok(MutexGuard::new(self)?)
            } else {
                Err(TryLockError::WouldBlock)
            }
        }
    }

    /// Determines whether the mutex is poisoned.
    ///
    /// If another thread is active, the mutex can still become poisoned at any
    /// time. You should not trust a `false` value for program correctness
    /// without additional synchronization.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::sync::{Arc, Mutex};
    /// use std::thread;
    ///
    /// let mutex = Arc::new(Mutex::new(0));
    /// let c_mutex = Arc::clone(&mutex);
    ///
    /// let _ = thread::spawn(move || {
    ///     let _lock = c_mutex.lock().unwrap();
    ///     panic!(); // the mutex gets poisoned
    /// }).join();
    /// assert_eq!(mutex.is_poisoned(), true);
    /// ```
    #[inline]
    #[stable(feature = "sync_poison", since = "1.2.0")]
    pub fn is_poisoned(&self) -> bool {
        self.poison.get()
    }

    /// Clear the poisoned state from a mutex.
    ///
    /// If the mutex is poisoned, it will remain poisoned until this function is called. This
    /// allows recovering from a poisoned state and marking that it has recovered. For example, if
    /// the value is overwritten by a known-good value, then the mutex can be marked as
    /// un-poisoned. Or possibly, the value could be inspected to determine if it is in a
    /// consistent state, and if so the poison is removed.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::sync::{Arc, Mutex};
    /// use std::thread;
    ///
    /// let mutex = Arc::new(Mutex::new(0));
    /// let c_mutex = Arc::clone(&mutex);
    ///
    /// let _ = thread::spawn(move || {
    ///     let _lock = c_mutex.lock().unwrap();
    ///     panic!(); // the mutex gets poisoned
    /// }).join();
    ///
    /// assert_eq!(mutex.is_poisoned(), true);
    /// let x = mutex.lock().unwrap_or_else(|mut e| {
    ///     **e.get_mut() = 1;
    ///     mutex.clear_poison();
    ///     e.into_inner()
    /// });
    /// assert_eq!(mutex.is_poisoned(), false);
    /// assert_eq!(*x, 1);
    /// ```
    #[inline]
    #[stable(feature = "mutex_unpoison", since = "1.77.0")]
    pub fn clear_poison(&self) {
        self.poison.clear();
    }

    /// Consumes this mutex, returning the underlying data.
    ///
    /// # Errors
    ///
    /// If another user of this mutex panicked while holding the mutex, then
    /// this call will return an error containing the underlying data
    /// instead.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::sync::Mutex;
    ///
    /// let mutex = Mutex::new(0);
    /// assert_eq!(mutex.into_inner().unwrap(), 0);
    /// ```
    #[stable(feature = "mutex_into_inner", since = "1.6.0")]
    pub fn into_inner(self) -> LockResult<T>
    where
        T: Sized,
    {
        let data = self.data.into_inner();
        poison::map_result(self.poison.borrow(), |()| data)
    }

    /// Returns a mutable reference to the underlying data.
    ///
    /// Since this call borrows the `Mutex` mutably, no actual locking needs to
    /// take place -- the mutable borrow statically guarantees no new locks can be acquired
    /// while this reference exists. Note that this method does not clear any previous abandoned locks
    /// (e.g., via [`forget()`] on a [`MutexGuard`]).
    ///
    /// # Errors
    ///
    /// If another user of this mutex panicked while holding the mutex, then
    /// this call will return an error containing a mutable reference to the
    /// underlying data instead.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::sync::Mutex;
    ///
    /// let mut mutex = Mutex::new(0);
    /// *mutex.get_mut().unwrap() = 10;
    /// assert_eq!(*mutex.lock().unwrap(), 10);
    /// ```
    ///
    /// [`forget()`]: mem::forget
    #[stable(feature = "mutex_get_mut", since = "1.6.0")]
    pub fn get_mut(&mut self) -> LockResult<&mut T> {
        let data = self.data.get_mut();
        poison::map_result(self.poison.borrow(), |()| data)
    }
}

#[stable(feature = "mutex_from", since = "1.24.0")]
impl<T> From<T> for Mutex<T> {
    /// Creates a new mutex in an unlocked state ready for use.
    /// This is equivalent to [`Mutex::new`].
    fn from(t: T) -> Self {
        Mutex::new(t)
    }
}

#[stable(feature = "mutex_default", since = "1.10.0")]
impl<T: ?Sized + Default> Default for Mutex<T> {
    /// Creates a `Mutex<T>`, with the `Default` value for T.
    fn default() -> Mutex<T> {
        Mutex::new(Default::default())
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<T: ?Sized + fmt::Debug> fmt::Debug for Mutex<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut d = f.debug_struct("Mutex");
        match self.try_lock() {
            Ok(guard) => {
                d.field("data", &&*guard);
            }
            Err(TryLockError::Poisoned(err)) => {
                d.field("data", &&**err.get_ref());
            }
            Err(TryLockError::WouldBlock) => {
                d.field("data", &format_args!("<locked>"));
            }
        }
        d.field("poisoned", &self.poison.get());
        d.finish_non_exhaustive()
    }
}

impl<'mutex, T: ?Sized> MutexGuard<'mutex, T> {
    unsafe fn new(lock: &'mutex Mutex<T>) -> LockResult<MutexGuard<'mutex, T>> {
        poison::map_result(lock.poison.guard(), |guard| MutexGuard { lock, poison: guard })
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<T: ?Sized> Deref for MutexGuard<'_, T> {
    type Target = T;

    fn deref(&self) -> &T {
        unsafe { &*self.lock.data.get() }
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<T: ?Sized> DerefMut for MutexGuard<'_, T> {
    fn deref_mut(&mut self) -> &mut T {
        unsafe { &mut *self.lock.data.get() }
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<T: ?Sized> Drop for MutexGuard<'_, T> {
    #[inline]
    fn drop(&mut self) {
        unsafe {
            self.lock.poison.done(&self.poison);
            self.lock.inner.unlock();
        }
    }
}

#[stable(feature = "std_debug", since = "1.16.0")]
impl<T: ?Sized + fmt::Debug> fmt::Debug for MutexGuard<'_, T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Debug::fmt(&**self, f)
    }
}

#[stable(feature = "std_guard_impls", since = "1.20.0")]
impl<T: ?Sized + fmt::Display> fmt::Display for MutexGuard<'_, T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        (**self).fmt(f)
    }
}

pub fn guard_lock<'a, T: ?Sized>(guard: &MutexGuard<'a, T>) -> &'a sys::Mutex {
    &guard.lock.inner
}

pub fn guard_poison<'a, T: ?Sized>(guard: &MutexGuard<'a, T>) -> &'a poison::Flag {
    &guard.lock.poison
}

impl<'a, T: ?Sized> MutexGuard<'a, T> {
    /// Makes a [`MappedMutexGuard`] for a component of the borrowed data, e.g.
    /// an enum variant.
    ///
    /// The `Mutex` is already locked, so this cannot fail.
    ///
    /// This is an associated function that needs to be used as
    /// `MutexGuard::map(...)`. A method would interfere with methods of the
    /// same name on the contents of the `MutexGuard` used through `Deref`.
    #[unstable(feature = "mapped_lock_guards", issue = "117108")]
    pub fn map<U, F>(orig: Self, f: F) -> MappedMutexGuard<'a, U>
    where
        F: FnOnce(&mut T) -> &mut U,
        U: ?Sized,
    {
        // SAFETY: the conditions of `MutexGuard::new` were satisfied when the original guard
        // was created, and have been upheld throughout `map` and/or `try_map`.
        // The signature of the closure guarantees that it will not "leak" the lifetime of the reference
        // passed to it. If the closure panics, the guard will be dropped.
        let data = NonNull::from(f(unsafe { &mut *orig.lock.data.get() }));
        let orig = ManuallyDrop::new(orig);
        MappedMutexGuard {
            data,
            inner: &orig.lock.inner,
            poison_flag: &orig.lock.poison,
            poison: orig.poison.clone(),
            _variance: PhantomData,
        }
    }

    /// Makes a [`MappedMutexGuard`] for a component of the borrowed data. The
    /// original guard is returned as an `Err(...)` if the closure returns
    /// `None`.
    ///
    /// The `Mutex` is already locked, so this cannot fail.
    ///
    /// This is an associated function that needs to be used as
    /// `MutexGuard::try_map(...)`. A method would interfere with methods of the
    /// same name on the contents of the `MutexGuard` used through `Deref`.
    #[doc(alias = "filter_map")]
    #[unstable(feature = "mapped_lock_guards", issue = "117108")]
    pub fn try_map<U, F>(orig: Self, f: F) -> Result<MappedMutexGuard<'a, U>, Self>
    where
        F: FnOnce(&mut T) -> Option<&mut U>,
        U: ?Sized,
    {
        // SAFETY: the conditions of `MutexGuard::new` were satisfied when the original guard
        // was created, and have been upheld throughout `map` and/or `try_map`.
        // The signature of the closure guarantees that it will not "leak" the lifetime of the reference
        // passed to it. If the closure panics, the guard will be dropped.
        match f(unsafe { &mut *orig.lock.data.get() }) {
            Some(data) => {
                let data = NonNull::from(data);
                let orig = ManuallyDrop::new(orig);
                Ok(MappedMutexGuard {
                    data,
                    inner: &orig.lock.inner,
                    poison_flag: &orig.lock.poison,
                    poison: orig.poison.clone(),
                    _variance: PhantomData,
                })
            }
            None => Err(orig),
        }
    }
}

#[unstable(feature = "mapped_lock_guards", issue = "117108")]
impl<T: ?Sized> Deref for MappedMutexGuard<'_, T> {
    type Target = T;

    fn deref(&self) -> &T {
        unsafe { self.data.as_ref() }
    }
}

#[unstable(feature = "mapped_lock_guards", issue = "117108")]
impl<T: ?Sized> DerefMut for MappedMutexGuard<'_, T> {
    fn deref_mut(&mut self) -> &mut T {
        unsafe { self.data.as_mut() }
    }
}

#[unstable(feature = "mapped_lock_guards", issue = "117108")]
impl<T: ?Sized> Drop for MappedMutexGuard<'_, T> {
    #[inline]
    fn drop(&mut self) {
        unsafe {
            self.poison_flag.done(&self.poison);
            self.inner.unlock();
        }
    }
}

#[unstable(feature = "mapped_lock_guards", issue = "117108")]
impl<T: ?Sized + fmt::Debug> fmt::Debug for MappedMutexGuard<'_, T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Debug::fmt(&**self, f)
    }
}

#[unstable(feature = "mapped_lock_guards", issue = "117108")]
impl<T: ?Sized + fmt::Display> fmt::Display for MappedMutexGuard<'_, T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        (**self).fmt(f)
    }
}

impl<'a, T: ?Sized> MappedMutexGuard<'a, T> {
    /// Makes a [`MappedMutexGuard`] for a component of the borrowed data, e.g.
    /// an enum variant.
    ///
    /// The `Mutex` is already locked, so this cannot fail.
    ///
    /// This is an associated function that needs to be used as
    /// `MappedMutexGuard::map(...)`. A method would interfere with methods of the
    /// same name on the contents of the `MutexGuard` used through `Deref`.
    #[unstable(feature = "mapped_lock_guards", issue = "117108")]
    pub fn map<U, F>(mut orig: Self, f: F) -> MappedMutexGuard<'a, U>
    where
        F: FnOnce(&mut T) -> &mut U,
        U: ?Sized,
    {
        // SAFETY: the conditions of `MutexGuard::new` were satisfied when the original guard
        // was created, and have been upheld throughout `map` and/or `try_map`.
        // The signature of the closure guarantees that it will not "leak" the lifetime of the reference
        // passed to it. If the closure panics, the guard will be dropped.
        let data = NonNull::from(f(unsafe { orig.data.as_mut() }));
        let orig = ManuallyDrop::new(orig);
        MappedMutexGuard {
            data,
            inner: orig.inner,
            poison_flag: orig.poison_flag,
            poison: orig.poison.clone(),
            _variance: PhantomData,
        }
    }

    /// Makes a [`MappedMutexGuard`] for a component of the borrowed data. The
    /// original guard is returned as an `Err(...)` if the closure returns
    /// `None`.
    ///
    /// The `Mutex` is already locked, so this cannot fail.
    ///
    /// This is an associated function that needs to be used as
    /// `MappedMutexGuard::try_map(...)`. A method would interfere with methods of the
    /// same name on the contents of the `MutexGuard` used through `Deref`.
    #[doc(alias = "filter_map")]
    #[unstable(feature = "mapped_lock_guards", issue = "117108")]
    pub fn try_map<U, F>(mut orig: Self, f: F) -> Result<MappedMutexGuard<'a, U>, Self>
    where
        F: FnOnce(&mut T) -> Option<&mut U>,
        U: ?Sized,
    {
        // SAFETY: the conditions of `MutexGuard::new` were satisfied when the original guard
        // was created, and have been upheld throughout `map` and/or `try_map`.
        // The signature of the closure guarantees that it will not "leak" the lifetime of the reference
        // passed to it. If the closure panics, the guard will be dropped.
        match f(unsafe { orig.data.as_mut() }) {
            Some(data) => {
                let data = NonNull::from(data);
                let orig = ManuallyDrop::new(orig);
                Ok(MappedMutexGuard {
                    data,
                    inner: orig.inner,
                    poison_flag: orig.poison_flag,
                    poison: orig.poison.clone(),
                    _variance: PhantomData,
                })
            }
            None => Err(orig),
        }
    }
}
