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
use sync::poison::{self, LockResult, TryLockError, TryLockResult};
use sys_common::rwlock as sys;

/// A reader-writer lock
///
/// This type of lock allows a number of readers or at most one writer at any
/// point in time. The write portion of this lock typically allows modification
/// of the underlying data (exclusive access) and the read portion of this lock
/// typically allows for read-only access (shared access).
///
/// The type parameter `T` represents the data that this lock protects. It is
/// required that `T` satisfies `Send` to be shared across tasks and `Sync` to
/// allow concurrent access through readers. The RAII guards returned from the
/// locking methods implement `Deref` (and `DerefMut` for the `write` methods)
/// to allow access to the contained of the lock.
///
/// # Poisoning
///
/// RwLocks, like Mutexes, will become poisoned on panics. Note, however, that
/// an RwLock may only be poisoned if a panic occurs while it is locked
/// exclusively (write mode). If a panic occurs in any reader, then the lock
/// will not be poisoned.
///
/// # Examples
///
/// ```
/// use std::sync::RwLock;
///
/// let lock = RwLock::new(5);
///
/// // many reader locks can be held at once
/// {
///     let r1 = lock.read().unwrap();
///     let r2 = lock.read().unwrap();
///     assert_eq!(*r1, 5);
///     assert_eq!(*r2, 5);
/// } // read locks are dropped at this point
///
/// // only one write lock may be held, however
/// {
///     let mut w = lock.write().unwrap();
///     *w += 1;
///     assert_eq!(*w, 6);
/// } // write lock is dropped here
/// ```
#[stable(feature = "rust1", since = "1.0.0")]
pub struct RwLock<T> {
    inner: Box<StaticRwLock>,
    data: UnsafeCell<T>,
}

unsafe impl<T:'static+Send> Send for RwLock<T> {}
unsafe impl<T> Sync for RwLock<T> {}

/// Structure representing a statically allocated RwLock.
///
/// This structure is intended to be used inside of a `static` and will provide
/// automatic global access as well as lazy initialization. The internal
/// resources of this RwLock, however, must be manually deallocated.
///
/// # Example
///
/// ```
/// use std::sync::{StaticRwLock, RW_LOCK_INIT};
///
/// static LOCK: StaticRwLock = RW_LOCK_INIT;
///
/// {
///     let _g = LOCK.read().unwrap();
///     // ... shared read access
/// }
/// {
///     let _g = LOCK.write().unwrap();
///     // ... exclusive write access
/// }
/// unsafe { LOCK.destroy() } // free all resources
/// ```
#[unstable(feature = "std_misc",
           reason = "may be merged with RwLock in the future")]
pub struct StaticRwLock {
    lock: sys::RWLock,
    poison: poison::Flag,
}

unsafe impl Send for StaticRwLock {}
unsafe impl Sync for StaticRwLock {}

/// Constant initialization for a statically-initialized rwlock.
#[unstable(feature = "std_misc",
           reason = "may be merged with RwLock in the future")]
pub const RW_LOCK_INIT: StaticRwLock = StaticRwLock {
    lock: sys::RWLOCK_INIT,
    poison: poison::FLAG_INIT,
};

/// RAII structure used to release the shared read access of a lock when
/// dropped.
#[must_use]
#[stable(feature = "rust1", since = "1.0.0")]
pub struct RwLockReadGuard<'a, T: 'a> {
    __lock: &'a StaticRwLock,
    __data: &'a UnsafeCell<T>,
}

impl<'a, T> !marker::Send for RwLockReadGuard<'a, T> {}

/// RAII structure used to release the exclusive write access of a lock when
/// dropped.
#[must_use]
#[stable(feature = "rust1", since = "1.0.0")]
pub struct RwLockWriteGuard<'a, T: 'a> {
    __lock: &'a StaticRwLock,
    __data: &'a UnsafeCell<T>,
    __poison: poison::Guard,
}

impl<'a, T> !marker::Send for RwLockWriteGuard<'a, T> {}

impl<T: Send + Sync> RwLock<T> {
    /// Creates a new instance of an `RwLock<T>` which is unlocked.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::sync::RwLock;
    ///
    /// let lock = RwLock::new(5);
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn new(t: T) -> RwLock<T> {
        RwLock { inner: box RW_LOCK_INIT, data: UnsafeCell::new(t) }
    }

    /// Locks this rwlock with shared read access, blocking the current thread
    /// until it can be acquired.
    ///
    /// The calling thread will be blocked until there are no more writers which
    /// hold the lock. There may be other readers currently inside the lock when
    /// this method returns. This method does not provide any guarantees with
    /// respect to the ordering of whether contentious readers or writers will
    /// acquire the lock first.
    ///
    /// Returns an RAII guard which will release this thread's shared access
    /// once it is dropped.
    ///
    /// # Failure
    ///
    /// This function will return an error if the RwLock is poisoned. An RwLock
    /// is poisoned whenever a writer panics while holding an exclusive lock.
    /// The failure will occur immediately after the lock has been acquired.
    #[inline]
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn read(&self) -> LockResult<RwLockReadGuard<T>> {
        unsafe { self.inner.lock.read() }
        RwLockReadGuard::new(&*self.inner, &self.data)
    }

    /// Attempt to acquire this lock with shared read access.
    ///
    /// This function will never block and will return immediately if `read`
    /// would otherwise succeed. Returns `Some` of an RAII guard which will
    /// release the shared access of this thread when dropped, or `None` if the
    /// access could not be granted. This method does not provide any
    /// guarantees with respect to the ordering of whether contentious readers
    /// or writers will acquire the lock first.
    ///
    /// # Failure
    ///
    /// This function will return an error if the RwLock is poisoned. An RwLock
    /// is poisoned whenever a writer panics while holding an exclusive lock. An
    /// error will only be returned if the lock would have otherwise been
    /// acquired.
    #[inline]
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn try_read(&self) -> TryLockResult<RwLockReadGuard<T>> {
        if unsafe { self.inner.lock.try_read() } {
            Ok(try!(RwLockReadGuard::new(&*self.inner, &self.data)))
        } else {
            Err(TryLockError::WouldBlock)
        }
    }

    /// Lock this rwlock with exclusive write access, blocking the current
    /// thread until it can be acquired.
    ///
    /// This function will not return while other writers or other readers
    /// currently have access to the lock.
    ///
    /// Returns an RAII guard which will drop the write access of this rwlock
    /// when dropped.
    ///
    /// # Failure
    ///
    /// This function will return an error if the RwLock is poisoned. An RwLock
    /// is poisoned whenever a writer panics while holding an exclusive lock.
    /// An error will be returned when the lock is acquired.
    #[inline]
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn write(&self) -> LockResult<RwLockWriteGuard<T>> {
        unsafe { self.inner.lock.write() }
        RwLockWriteGuard::new(&*self.inner, &self.data)
    }

    /// Attempt to lock this rwlock with exclusive write access.
    ///
    /// This function does not ever block, and it will return `None` if a call
    /// to `write` would otherwise block. If successful, an RAII guard is
    /// returned.
    ///
    /// # Failure
    ///
    /// This function will return an error if the RwLock is poisoned. An RwLock
    /// is poisoned whenever a writer panics while holding an exclusive lock. An
    /// error will only be returned if the lock would have otherwise been
    /// acquired.
    #[inline]
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn try_write(&self) -> TryLockResult<RwLockWriteGuard<T>> {
        if unsafe { self.inner.lock.try_read() } {
            Ok(try!(RwLockWriteGuard::new(&*self.inner, &self.data)))
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
impl<T> Drop for RwLock<T> {
    fn drop(&mut self) {
        unsafe { self.inner.lock.destroy() }
    }
}

struct Dummy(UnsafeCell<()>);
unsafe impl Sync for Dummy {}
static DUMMY: Dummy = Dummy(UnsafeCell { value: () });

impl StaticRwLock {
    /// Locks this rwlock with shared read access, blocking the current thread
    /// until it can be acquired.
    ///
    /// See `RwLock::read`.
    #[inline]
    #[unstable(feature = "std_misc",
               reason = "may be merged with RwLock in the future")]
    pub fn read(&'static self) -> LockResult<RwLockReadGuard<'static, ()>> {
        unsafe { self.lock.read() }
        RwLockReadGuard::new(self, &DUMMY.0)
    }

    /// Attempt to acquire this lock with shared read access.
    ///
    /// See `RwLock::try_read`.
    #[inline]
    #[unstable(feature = "std_misc",
               reason = "may be merged with RwLock in the future")]
    pub fn try_read(&'static self)
                    -> TryLockResult<RwLockReadGuard<'static, ()>> {
        if unsafe { self.lock.try_read() } {
            Ok(try!(RwLockReadGuard::new(self, &DUMMY.0)))
        } else {
            Err(TryLockError::WouldBlock)
        }
    }

    /// Lock this rwlock with exclusive write access, blocking the current
    /// thread until it can be acquired.
    ///
    /// See `RwLock::write`.
    #[inline]
    #[unstable(feature = "std_misc",
               reason = "may be merged with RwLock in the future")]
    pub fn write(&'static self) -> LockResult<RwLockWriteGuard<'static, ()>> {
        unsafe { self.lock.write() }
        RwLockWriteGuard::new(self, &DUMMY.0)
    }

    /// Attempt to lock this rwlock with exclusive write access.
    ///
    /// See `RwLock::try_write`.
    #[inline]
    #[unstable(feature = "std_misc",
               reason = "may be merged with RwLock in the future")]
    pub fn try_write(&'static self)
                     -> TryLockResult<RwLockWriteGuard<'static, ()>> {
        if unsafe { self.lock.try_write() } {
            Ok(try!(RwLockWriteGuard::new(self, &DUMMY.0)))
        } else {
            Err(TryLockError::WouldBlock)
        }
    }

    /// Deallocate all resources associated with this static lock.
    ///
    /// This method is unsafe to call as there is no guarantee that there are no
    /// active users of the lock, and this also doesn't prevent any future users
    /// of this lock. This method is required to be called to not leak memory on
    /// all platforms.
    #[unstable(feature = "std_misc",
               reason = "may be merged with RwLock in the future")]
    pub unsafe fn destroy(&'static self) {
        self.lock.destroy()
    }
}

impl<'rwlock, T> RwLockReadGuard<'rwlock, T> {

    fn new(lock: &'rwlock StaticRwLock, data: &'rwlock UnsafeCell<T>)
           -> LockResult<RwLockReadGuard<'rwlock, T>> {
        poison::map_result(lock.poison.borrow(), |_| {
            RwLockReadGuard {
                __lock: lock,
                __data: data,
            }
        })
    }
}
impl<'rwlock, T> RwLockWriteGuard<'rwlock, T> {

    fn new(lock: &'rwlock StaticRwLock, data: &'rwlock UnsafeCell<T>)
           -> LockResult<RwLockWriteGuard<'rwlock, T>> {
        poison::map_result(lock.poison.borrow(), |guard| {
            RwLockWriteGuard {
                __lock: lock,
                __data: data,
                __poison: guard,
            }
        })
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<'rwlock, T> Deref for RwLockReadGuard<'rwlock, T> {
    type Target = T;

    fn deref(&self) -> &T { unsafe { &*self.__data.get() } }
}
#[stable(feature = "rust1", since = "1.0.0")]
impl<'rwlock, T> Deref for RwLockWriteGuard<'rwlock, T> {
    type Target = T;

    fn deref(&self) -> &T { unsafe { &*self.__data.get() } }
}
#[stable(feature = "rust1", since = "1.0.0")]
impl<'rwlock, T> DerefMut for RwLockWriteGuard<'rwlock, T> {
    fn deref_mut(&mut self) -> &mut T {
        unsafe { &mut *self.__data.get() }
    }
}

#[unsafe_destructor]
#[stable(feature = "rust1", since = "1.0.0")]
impl<'a, T> Drop for RwLockReadGuard<'a, T> {
    fn drop(&mut self) {
        unsafe { self.__lock.lock.read_unlock(); }
    }
}

#[unsafe_destructor]
#[stable(feature = "rust1", since = "1.0.0")]
impl<'a, T> Drop for RwLockWriteGuard<'a, T> {
    fn drop(&mut self) {
        self.__lock.poison.done(&self.__poison);
        unsafe { self.__lock.lock.write_unlock(); }
    }
}

#[cfg(test)]
mod tests {
    #![allow(deprecated)] // rand

    use prelude::v1::*;

    use rand::{self, Rng};
    use sync::mpsc::channel;
    use thread;
    use sync::{Arc, RwLock, StaticRwLock, RW_LOCK_INIT};

    #[test]
    fn smoke() {
        let l = RwLock::new(());
        drop(l.read().unwrap());
        drop(l.write().unwrap());
        drop((l.read().unwrap(), l.read().unwrap()));
        drop(l.write().unwrap());
    }

    #[test]
    fn static_smoke() {
        static R: StaticRwLock = RW_LOCK_INIT;
        drop(R.read().unwrap());
        drop(R.write().unwrap());
        drop((R.read().unwrap(), R.read().unwrap()));
        drop(R.write().unwrap());
        unsafe { R.destroy(); }
    }

    #[test]
    fn frob() {
        static R: StaticRwLock = RW_LOCK_INIT;
        static N: usize = 10;
        static M: usize = 1000;

        let (tx, rx) = channel::<()>();
        for _ in 0..N {
            let tx = tx.clone();
            thread::spawn(move|| {
                let mut rng = rand::thread_rng();
                for _ in 0..M {
                    if rng.gen_weighted_bool(N) {
                        drop(R.write().unwrap());
                    } else {
                        drop(R.read().unwrap());
                    }
                }
                drop(tx);
            });
        }
        drop(tx);
        let _ = rx.recv();
        unsafe { R.destroy(); }
    }

    #[test]
    fn test_rw_arc_poison_wr() {
        let arc = Arc::new(RwLock::new(1));
        let arc2 = arc.clone();
        let _: Result<(), _> = thread::spawn(move|| {
            let _lock = arc2.write().unwrap();
            panic!();
        }).join();
        assert!(arc.read().is_err());
    }

    #[test]
    fn test_rw_arc_poison_ww() {
        let arc = Arc::new(RwLock::new(1));
        assert!(!arc.is_poisoned());
        let arc2 = arc.clone();
        let _: Result<(), _> = thread::spawn(move|| {
            let _lock = arc2.write().unwrap();
            panic!();
        }).join();
        assert!(arc.write().is_err());
        assert!(arc.is_poisoned());
    }

    #[test]
    fn test_rw_arc_no_poison_rr() {
        let arc = Arc::new(RwLock::new(1));
        let arc2 = arc.clone();
        let _: Result<(), _> = thread::spawn(move|| {
            let _lock = arc2.read().unwrap();
            panic!();
        }).join();
        let lock = arc.read().unwrap();
        assert_eq!(*lock, 1);
    }
    #[test]
    fn test_rw_arc_no_poison_rw() {
        let arc = Arc::new(RwLock::new(1));
        let arc2 = arc.clone();
        let _: Result<(), _> = thread::spawn(move|| {
            let _lock = arc2.read().unwrap();
            panic!()
        }).join();
        let lock = arc.write().unwrap();
        assert_eq!(*lock, 1);
    }

    #[test]
    fn test_rw_arc() {
        let arc = Arc::new(RwLock::new(0));
        let arc2 = arc.clone();
        let (tx, rx) = channel();

        thread::spawn(move|| {
            let mut lock = arc2.write().unwrap();
            for _ in 0..10 {
                let tmp = *lock;
                *lock = -1;
                thread::yield_now();
                *lock = tmp + 1;
            }
            tx.send(()).unwrap();
        });

        // Readers try to catch the writer in the act
        let mut children = Vec::new();
        for _ in 0..5 {
            let arc3 = arc.clone();
            children.push(thread::spawn(move|| {
                let lock = arc3.read().unwrap();
                assert!(*lock >= 0);
            }));
        }

        // Wait for children to pass their asserts
        for r in children {
            assert!(r.join().is_ok());
        }

        // Wait for writer to finish
        rx.recv().unwrap();
        let lock = arc.read().unwrap();
        assert_eq!(*lock, 10);
    }

    #[test]
    fn test_rw_arc_access_in_unwind() {
        let arc = Arc::new(RwLock::new(1));
        let arc2 = arc.clone();
        let _ = thread::spawn(move|| -> () {
            struct Unwinder {
                i: Arc<RwLock<int>>,
            }
            impl Drop for Unwinder {
                fn drop(&mut self) {
                    let mut lock = self.i.write().unwrap();
                    *lock += 1;
                }
            }
            let _u = Unwinder { i: arc2 };
            panic!();
        }).join();
        let lock = arc.read().unwrap();
        assert_eq!(*lock, 2);
    }
}
