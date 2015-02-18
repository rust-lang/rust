// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use prelude::v1::*;

use cell::UnsafeCell;
use marker;
use ops::{Deref, DerefMut};
use sync::poison::{self, TryLockError, TryLockResult, LockResult};
use sys_common::mutex as sys;

/// A mutual exclusion primitive useful for protecting shared data
///
/// This mutex will block threads waiting for the lock to become available. The
/// mutex can also be statically initialized or created via a `new`
/// constructor. Each mutex has a type parameter which represents the data that
/// it is protecting. The data can only be accessed through the RAII guards
/// returned from `lock` and `try_lock`, which guarantees that the data is only
/// ever accessed when the mutex is locked.
///
/// # Poisoning
///
/// The mutexes in this module implement a strategy called "poisoning" where a
/// mutex is considered poisoned whenever a thread panics while holding the
/// lock. Once a mutex is poisoned, all other tasks are unable to access the
/// data by default as it is likely tainted (some invariant is not being
/// upheld).
///
/// For a mutex, this means that the `lock` and `try_lock` methods return a
/// `Result` which indicates whether a mutex has been poisoned or not. Most
/// usage of a mutex will simply `unwrap()` these results, propagating panics
/// among threads to ensure that a possibly invalid invariant is not witnessed.
///
/// A poisoned mutex, however, does not prevent all access to the underlying
/// data. The `PoisonError` type has an `into_guard` method which will return
/// the guard that would have otherwise been returned on a successful lock. This
/// allows access to the data, despite the lock being poisoned.
///
/// # Examples
///
/// ```rust
/// use std::sync::{Arc, Mutex};
/// use std::thread;
/// use std::sync::mpsc::channel;
///
/// const N: uint = 10;
///
/// // Spawn a few threads to increment a shared variable (non-atomically), and
/// // let the main thread know once all increments are done.
/// //
/// // Here we're using an Arc to share memory among tasks, and the data inside
/// // the Arc is protected with a mutex.
/// let data = Arc::new(Mutex::new(0));
///
/// let (tx, rx) = channel();
/// for _ in 0..10 {
///     let (data, tx) = (data.clone(), tx.clone());
///     thread::spawn(move || {
///         // The shared static can only be accessed once the lock is held.
///         // Our non-atomic increment is safe because we're the only thread
///         // which can access the shared state when the lock is held.
///         //
///         // We unwrap() the return value to assert that we are not expecting
///         // tasks to ever fail while holding the lock.
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
/// ```rust
/// use std::sync::{Arc, Mutex};
/// use std::thread;
///
/// let lock = Arc::new(Mutex::new(0_u32));
/// let lock2 = lock.clone();
///
/// let _ = thread::spawn(move || -> () {
///     // This thread will acquire the mutex first, unwrapping the result of
///     // `lock` because the lock has not been poisoned.
///     let _lock = lock2.lock().unwrap();
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
///     Err(poisoned) => poisoned.into_guard(),
/// };
///
/// *guard += 1;
/// ```
#[stable(feature = "rust1", since = "1.0.0")]
pub struct Mutex<T> {
    // Note that this static mutex is in a *box*, not inlined into the struct
    // itself. Once a native mutex has been used once, its address can never
    // change (it can't be moved). This mutex type can be safely moved at any
    // time, so to ensure that the native mutex is used correctly we box the
    // inner lock to give it a constant address.
    inner: Box<StaticMutex>,
    data: UnsafeCell<T>,
}

unsafe impl<T: Send + 'static> Send for Mutex<T> { }

unsafe impl<T: Send + 'static> Sync for Mutex<T> { }

/// The static mutex type is provided to allow for static allocation of mutexes.
///
/// Note that this is a separate type because using a Mutex correctly means that
/// it needs to have a destructor run. In Rust, statics are not allowed to have
/// destructors. As a result, a `StaticMutex` has one extra method when compared
/// to a `Mutex`, a `destroy` method. This method is unsafe to call, and
/// documentation can be found directly on the method.
///
/// # Example
///
/// ```rust
/// use std::sync::{StaticMutex, MUTEX_INIT};
///
/// static LOCK: StaticMutex = MUTEX_INIT;
///
/// {
///     let _g = LOCK.lock().unwrap();
///     // do some productive work
/// }
/// // lock is unlocked here.
/// ```
#[unstable(feature = "std_misc",
           reason = "may be merged with Mutex in the future")]
pub struct StaticMutex {
    lock: sys::Mutex,
    poison: poison::Flag,
}

unsafe impl Sync for StaticMutex {}

/// An RAII implementation of a "scoped lock" of a mutex. When this structure is
/// dropped (falls out of scope), the lock will be unlocked.
///
/// The data protected by the mutex can be access through this guard via its
/// Deref and DerefMut implementations
#[must_use]
#[stable(feature = "rust1", since = "1.0.0")]
pub struct MutexGuard<'a, T: 'a> {
    // funny underscores due to how Deref/DerefMut currently work (they
    // disregard field privacy).
    __lock: &'a StaticMutex,
    __data: &'a UnsafeCell<T>,
    __poison: poison::Guard,
}

impl<'a, T> !marker::Send for MutexGuard<'a, T> {}

/// Static initialization of a mutex. This constant can be used to initialize
/// other mutex constants.
#[unstable(feature = "std_misc",
           reason = "may be merged with Mutex in the future")]
pub const MUTEX_INIT: StaticMutex = StaticMutex {
    lock: sys::MUTEX_INIT,
    poison: poison::FLAG_INIT,
};

impl<T: Send + 'static> Mutex<T> {
    /// Creates a new mutex in an unlocked state ready for use.
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn new(t: T) -> Mutex<T> {
        Mutex {
            inner: box MUTEX_INIT,
            data: UnsafeCell::new(t),
        }
    }

    /// Acquires a mutex, blocking the current task until it is able to do so.
    ///
    /// This function will block the local task until it is available to acquire
    /// the mutex. Upon returning, the task is the only task with the mutex
    /// held. An RAII guard is returned to allow scoped unlock of the lock. When
    /// the guard goes out of scope, the mutex will be unlocked.
    ///
    /// # Failure
    ///
    /// If another user of this mutex panicked while holding the mutex, then
    /// this call will return an error once the mutex is acquired.
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn lock(&self) -> LockResult<MutexGuard<T>> {
        unsafe { self.inner.lock.lock() }
        MutexGuard::new(&*self.inner, &self.data)
    }

    /// Attempts to acquire this lock.
    ///
    /// If the lock could not be acquired at this time, then `None` is returned.
    /// Otherwise, an RAII guard is returned. The lock will be unlocked when the
    /// guard is dropped.
    ///
    /// This function does not block.
    ///
    /// # Failure
    ///
    /// If another user of this mutex panicked while holding the mutex, then
    /// this call will return failure if the mutex would otherwise be
    /// acquired.
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn try_lock(&self) -> TryLockResult<MutexGuard<T>> {
        if unsafe { self.inner.lock.try_lock() } {
            Ok(try!(MutexGuard::new(&*self.inner, &self.data)))
        } else {
            Err(TryLockError::WouldBlock)
        }
    }

    /// Determine whether the lock is poisoned.
    ///
    /// If another thread is active, the lock can still become poisoned at any
    /// time.  You should not trust a `false` value for program correctness
    /// without additional synchronization.
    #[inline]
    #[unstable(feature = "std_misc")]
    pub fn is_poisoned(&self) -> bool {
        self.inner.poison.get()
    }
}

#[unsafe_destructor]
#[stable(feature = "rust1", since = "1.0.0")]
impl<T: Send + 'static> Drop for Mutex<T> {
    fn drop(&mut self) {
        // This is actually safe b/c we know that there is no further usage of
        // this mutex (it's up to the user to arrange for a mutex to get
        // dropped, that's not our job)
        unsafe { self.inner.lock.destroy() }
    }
}

struct Dummy(UnsafeCell<()>);
unsafe impl Sync for Dummy {}
static DUMMY: Dummy = Dummy(UnsafeCell { value: () });

impl StaticMutex {
    /// Acquires this lock, see `Mutex::lock`
    #[inline]
    #[unstable(feature = "std_misc",
               reason = "may be merged with Mutex in the future")]
    pub fn lock(&'static self) -> LockResult<MutexGuard<()>> {
        unsafe { self.lock.lock() }
        MutexGuard::new(self, &DUMMY.0)
    }

    /// Attempts to grab this lock, see `Mutex::try_lock`
    #[inline]
    #[unstable(feature = "std_misc",
               reason = "may be merged with Mutex in the future")]
    pub fn try_lock(&'static self) -> TryLockResult<MutexGuard<()>> {
        if unsafe { self.lock.try_lock() } {
            Ok(try!(MutexGuard::new(self, &DUMMY.0)))
        } else {
            Err(TryLockError::WouldBlock)
        }
    }

    /// Deallocates resources associated with this static mutex.
    ///
    /// This method is unsafe because it provides no guarantees that there are
    /// no active users of this mutex, and safety is not guaranteed if there are
    /// active users of this mutex.
    ///
    /// This method is required to ensure that there are no memory leaks on
    /// *all* platforms. It may be the case that some platforms do not leak
    /// memory if this method is not called, but this is not guaranteed to be
    /// true on all platforms.
    #[unstable(feature = "std_misc",
               reason = "may be merged with Mutex in the future")]
    pub unsafe fn destroy(&'static self) {
        self.lock.destroy()
    }
}

impl<'mutex, T> MutexGuard<'mutex, T> {

    fn new(lock: &'mutex StaticMutex, data: &'mutex UnsafeCell<T>)
           -> LockResult<MutexGuard<'mutex, T>> {
        poison::map_result(lock.poison.borrow(), |guard| {
            MutexGuard {
                __lock: lock,
                __data: data,
                __poison: guard,
            }
        })
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<'mutex, T> Deref for MutexGuard<'mutex, T> {
    type Target = T;

    fn deref<'a>(&'a self) -> &'a T {
        unsafe { &*self.__data.get() }
    }
}
#[stable(feature = "rust1", since = "1.0.0")]
impl<'mutex, T> DerefMut for MutexGuard<'mutex, T> {
    fn deref_mut<'a>(&'a mut self) -> &'a mut T {
        unsafe { &mut *self.__data.get() }
    }
}

#[unsafe_destructor]
#[stable(feature = "rust1", since = "1.0.0")]
impl<'a, T> Drop for MutexGuard<'a, T> {
    #[inline]
    fn drop(&mut self) {
        unsafe {
            self.__lock.poison.done(&self.__poison);
            self.__lock.lock.unlock();
        }
    }
}

pub fn guard_lock<'a, T>(guard: &MutexGuard<'a, T>) -> &'a sys::Mutex {
    &guard.__lock.lock
}

pub fn guard_poison<'a, T>(guard: &MutexGuard<'a, T>) -> &'a poison::Flag {
    &guard.__lock.poison
}

#[cfg(test)]
mod test {
    use prelude::v1::*;

    use sync::mpsc::channel;
    use sync::{Arc, Mutex, StaticMutex, MUTEX_INIT, Condvar};
    use thread;

    struct Packet<T>(Arc<(Mutex<T>, Condvar)>);

    unsafe impl<T:'static+Send> Send for Packet<T> {}
    unsafe impl<T> Sync for Packet<T> {}

    #[test]
    fn smoke() {
        let m = Mutex::new(());
        drop(m.lock().unwrap());
        drop(m.lock().unwrap());
    }

    #[test]
    fn smoke_static() {
        static M: StaticMutex = MUTEX_INIT;
        unsafe {
            drop(M.lock().unwrap());
            drop(M.lock().unwrap());
            M.destroy();
        }
    }

    #[test]
    fn lots_and_lots() {
        static M: StaticMutex = MUTEX_INIT;
        static mut CNT: uint = 0;
        static J: uint = 1000;
        static K: uint = 3;

        fn inc() {
            for _ in 0..J {
                unsafe {
                    let _g = M.lock().unwrap();
                    CNT += 1;
                }
            }
        }

        let (tx, rx) = channel();
        for _ in 0..K {
            let tx2 = tx.clone();
            thread::spawn(move|| { inc(); tx2.send(()).unwrap(); });
            let tx2 = tx.clone();
            thread::spawn(move|| { inc(); tx2.send(()).unwrap(); });
        }

        drop(tx);
        for _ in 0..2 * K {
            rx.recv().unwrap();
        }
        assert_eq!(unsafe {CNT}, J * K * 2);
        unsafe {
            M.destroy();
        }
    }

    #[test]
    fn try_lock() {
        let m = Mutex::new(());
        *m.try_lock().unwrap() = ();
    }

    #[test]
    fn test_mutex_arc_condvar() {
        let packet = Packet(Arc::new((Mutex::new(false), Condvar::new())));
        let packet2 = Packet(packet.0.clone());
        let (tx, rx) = channel();
        let _t = thread::spawn(move|| {
            // wait until parent gets in
            rx.recv().unwrap();
            let &(ref lock, ref cvar) = &*packet2.0;
            let mut lock = lock.lock().unwrap();
            *lock = true;
            cvar.notify_one();
        });

        let &(ref lock, ref cvar) = &*packet.0;
        let mut lock = lock.lock().unwrap();
        tx.send(()).unwrap();
        assert!(!*lock);
        while !*lock {
            lock = cvar.wait(lock).unwrap();
        }
    }

    #[test]
    fn test_arc_condvar_poison() {
        let packet = Packet(Arc::new((Mutex::new(1), Condvar::new())));
        let packet2 = Packet(packet.0.clone());
        let (tx, rx) = channel();

        let _t = thread::spawn(move || -> () {
            rx.recv().unwrap();
            let &(ref lock, ref cvar) = &*packet2.0;
            let _g = lock.lock().unwrap();
            cvar.notify_one();
            // Parent should fail when it wakes up.
            panic!();
        });

        let &(ref lock, ref cvar) = &*packet.0;
        let mut lock = lock.lock().unwrap();
        tx.send(()).unwrap();
        while *lock == 1 {
            match cvar.wait(lock) {
                Ok(l) => {
                    lock = l;
                    assert_eq!(*lock, 1);
                }
                Err(..) => break,
            }
        }
    }

    #[test]
    fn test_mutex_arc_poison() {
        let arc = Arc::new(Mutex::new(1));
        assert!(!arc.is_poisoned());
        let arc2 = arc.clone();
        let _ = thread::spawn(move|| {
            let lock = arc2.lock().unwrap();
            assert_eq!(*lock, 2);
        }).join();
        assert!(arc.lock().is_err());
        assert!(arc.is_poisoned());
    }

    #[test]
    fn test_mutex_arc_nested() {
        // Tests nested mutexes and access
        // to underlying data.
        let arc = Arc::new(Mutex::new(1));
        let arc2 = Arc::new(Mutex::new(arc));
        let (tx, rx) = channel();
        let _t = thread::spawn(move|| {
            let lock = arc2.lock().unwrap();
            let lock2 = lock.lock().unwrap();
            assert_eq!(*lock2, 1);
            tx.send(()).unwrap();
        });
        rx.recv().unwrap();
    }

    #[test]
    fn test_mutex_arc_access_in_unwind() {
        let arc = Arc::new(Mutex::new(1));
        let arc2 = arc.clone();
        let _ = thread::spawn(move|| -> () {
            struct Unwinder {
                i: Arc<Mutex<int>>,
            }
            impl Drop for Unwinder {
                fn drop(&mut self) {
                    *self.i.lock().unwrap() += 1;
                }
            }
            let _u = Unwinder { i: arc2 };
            panic!();
        }).join();
        let lock = arc.lock().unwrap();
        assert_eq!(*lock, 2);
    }
}
