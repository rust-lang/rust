// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use prelude::*;

use cell::UnsafeCell;
use kinds::marker;
use sync::poison::{mod, LockResult, TryLockError, TryLockResult};
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
/// RWLocks, like Mutexes, will become poisoned on panics. Note, however, that
/// an RWLock may only be poisoned if a panic occurs while it is locked
/// exclusively (write mode). If a panic occurs in any reader, then the lock
/// will not be poisoned.
///
/// # Examples
///
/// ```
/// use std::sync::RWLock;
///
/// let lock = RWLock::new(5i);
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
#[stable]
pub struct RWLock<T> {
    inner: Box<StaticRWLock>,
    data: UnsafeCell<T>,
}

unsafe impl<T:'static+Send> Send for RWLock<T> {}
unsafe impl<T> Sync for RWLock<T> {}

/// Structure representing a statically allocated RWLock.
///
/// This structure is intended to be used inside of a `static` and will provide
/// automatic global access as well as lazy initialization. The internal
/// resources of this RWLock, however, must be manually deallocated.
///
/// # Example
///
/// ```
/// use std::sync::{StaticRWLock, RWLOCK_INIT};
///
/// static LOCK: StaticRWLock = RWLOCK_INIT;
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
#[unstable = "may be merged with RWLock in the future"]
pub struct StaticRWLock {
    lock: sys::RWLock,
    poison: poison::Flag,
}

unsafe impl Send for StaticRWLock {}
unsafe impl Sync for StaticRWLock {}

/// Constant initialization for a statically-initialized rwlock.
#[unstable = "may be merged with RWLock in the future"]
pub const RWLOCK_INIT: StaticRWLock = StaticRWLock {
    lock: sys::RWLOCK_INIT,
    poison: poison::FLAG_INIT,
};

/// RAII structure used to release the shared read access of a lock when
/// dropped.
#[must_use]
#[stable]
pub struct RWLockReadGuard<'a, T: 'a> {
    __lock: &'a StaticRWLock,
    __data: &'a UnsafeCell<T>,
    __marker: marker::NoSend,
}

/// RAII structure used to release the exclusive write access of a lock when
/// dropped.
#[must_use]
#[stable]
pub struct RWLockWriteGuard<'a, T: 'a> {
    __lock: &'a StaticRWLock,
    __data: &'a UnsafeCell<T>,
    __poison: poison::Guard,
    __marker: marker::NoSend,
}

impl<T: Send + Sync> RWLock<T> {
    /// Creates a new instance of an RWLock which is unlocked and read to go.
    #[stable]
    pub fn new(t: T) -> RWLock<T> {
        RWLock { inner: box RWLOCK_INIT, data: UnsafeCell::new(t) }
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
    /// This function will return an error if the RWLock is poisoned. An RWLock
    /// is poisoned whenever a writer panics while holding an exclusive lock.
    /// The failure will occur immediately after the lock has been acquired.
    #[inline]
    #[stable]
    pub fn read(&self) -> LockResult<RWLockReadGuard<T>> {
        unsafe { self.inner.lock.read() }
        RWLockReadGuard::new(&*self.inner, &self.data)
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
    /// This function will return an error if the RWLock is poisoned. An RWLock
    /// is poisoned whenever a writer panics while holding an exclusive lock. An
    /// error will only be returned if the lock would have otherwise been
    /// acquired.
    #[inline]
    #[stable]
    pub fn try_read(&self) -> TryLockResult<RWLockReadGuard<T>> {
        if unsafe { self.inner.lock.try_read() } {
            Ok(try!(RWLockReadGuard::new(&*self.inner, &self.data)))
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
    /// This function will return an error if the RWLock is poisoned. An RWLock
    /// is poisoned whenever a writer panics while holding an exclusive lock.
    /// An error will be returned when the lock is acquired.
    #[inline]
    #[stable]
    pub fn write(&self) -> LockResult<RWLockWriteGuard<T>> {
        unsafe { self.inner.lock.write() }
        RWLockWriteGuard::new(&*self.inner, &self.data)
    }

    /// Attempt to lock this rwlock with exclusive write access.
    ///
    /// This function does not ever block, and it will return `None` if a call
    /// to `write` would otherwise block. If successful, an RAII guard is
    /// returned.
    ///
    /// # Failure
    ///
    /// This function will return an error if the RWLock is poisoned. An RWLock
    /// is poisoned whenever a writer panics while holding an exclusive lock. An
    /// error will only be returned if the lock would have otherwise been
    /// acquired.
    #[inline]
    #[stable]
    pub fn try_write(&self) -> TryLockResult<RWLockWriteGuard<T>> {
        if unsafe { self.inner.lock.try_read() } {
            Ok(try!(RWLockWriteGuard::new(&*self.inner, &self.data)))
        } else {
            Err(TryLockError::WouldBlock)
        }
    }
}

#[unsafe_destructor]
impl<T> Drop for RWLock<T> {
    fn drop(&mut self) {
        unsafe { self.inner.lock.destroy() }
    }
}

struct Dummy(UnsafeCell<()>);
unsafe impl Sync for Dummy {}
static DUMMY: Dummy = Dummy(UnsafeCell { value: () });

impl StaticRWLock {
    /// Locks this rwlock with shared read access, blocking the current thread
    /// until it can be acquired.
    ///
    /// See `RWLock::read`.
    #[inline]
    #[unstable = "may be merged with RWLock in the future"]
    pub fn read(&'static self) -> LockResult<RWLockReadGuard<'static, ()>> {
        unsafe { self.lock.read() }
        RWLockReadGuard::new(self, &DUMMY.0)
    }

    /// Attempt to acquire this lock with shared read access.
    ///
    /// See `RWLock::try_read`.
    #[inline]
    #[unstable = "may be merged with RWLock in the future"]
    pub fn try_read(&'static self)
                    -> TryLockResult<RWLockReadGuard<'static, ()>> {
        if unsafe { self.lock.try_read() } {
            Ok(try!(RWLockReadGuard::new(self, &DUMMY.0)))
        } else {
            Err(TryLockError::WouldBlock)
        }
    }

    /// Lock this rwlock with exclusive write access, blocking the current
    /// thread until it can be acquired.
    ///
    /// See `RWLock::write`.
    #[inline]
    #[unstable = "may be merged with RWLock in the future"]
    pub fn write(&'static self) -> LockResult<RWLockWriteGuard<'static, ()>> {
        unsafe { self.lock.write() }
        RWLockWriteGuard::new(self, &DUMMY.0)
    }

    /// Attempt to lock this rwlock with exclusive write access.
    ///
    /// See `RWLock::try_write`.
    #[inline]
    #[unstable = "may be merged with RWLock in the future"]
    pub fn try_write(&'static self)
                     -> TryLockResult<RWLockWriteGuard<'static, ()>> {
        if unsafe { self.lock.try_write() } {
            Ok(try!(RWLockWriteGuard::new(self, &DUMMY.0)))
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
    #[unstable = "may be merged with RWLock in the future"]
    pub unsafe fn destroy(&'static self) {
        self.lock.destroy()
    }
}

impl<'rwlock, T> RWLockReadGuard<'rwlock, T> {
    fn new(lock: &'rwlock StaticRWLock, data: &'rwlock UnsafeCell<T>)
           -> LockResult<RWLockReadGuard<'rwlock, T>> {
        poison::map_result(lock.poison.borrow(), |_| {
            RWLockReadGuard {
                __lock: lock,
                __data: data,
                __marker: marker::NoSend,
            }
        })
    }
}
impl<'rwlock, T> RWLockWriteGuard<'rwlock, T> {
    fn new(lock: &'rwlock StaticRWLock, data: &'rwlock UnsafeCell<T>)
           -> LockResult<RWLockWriteGuard<'rwlock, T>> {
        poison::map_result(lock.poison.borrow(), |guard| {
            RWLockWriteGuard {
                __lock: lock,
                __data: data,
                __poison: guard,
                __marker: marker::NoSend,
            }
        })
    }
}

impl<'rwlock, T> Deref<T> for RWLockReadGuard<'rwlock, T> {
    fn deref(&self) -> &T { unsafe { &*self.__data.get() } }
}
impl<'rwlock, T> Deref<T> for RWLockWriteGuard<'rwlock, T> {
    fn deref(&self) -> &T { unsafe { &*self.__data.get() } }
}
impl<'rwlock, T> DerefMut<T> for RWLockWriteGuard<'rwlock, T> {
    fn deref_mut(&mut self) -> &mut T {
        unsafe { &mut *self.__data.get() }
    }
}

#[unsafe_destructor]
impl<'a, T> Drop for RWLockReadGuard<'a, T> {
    fn drop(&mut self) {
        unsafe { self.__lock.lock.read_unlock(); }
    }
}

#[unsafe_destructor]
impl<'a, T> Drop for RWLockWriteGuard<'a, T> {
    fn drop(&mut self) {
        self.__lock.poison.done(&self.__poison);
        unsafe { self.__lock.lock.write_unlock(); }
    }
}

#[cfg(test)]
mod tests {
    use prelude::*;

    use rand::{mod, Rng};
    use thread::Thread;
    use sync::{Arc, RWLock, StaticRWLock, RWLOCK_INIT};

    #[test]
    fn smoke() {
        let l = RWLock::new(());
        drop(l.read().unwrap());
        drop(l.write().unwrap());
        drop((l.read().unwrap(), l.read().unwrap()));
        drop(l.write().unwrap());
    }

    #[test]
    fn static_smoke() {
        static R: StaticRWLock = RWLOCK_INIT;
        drop(R.read().unwrap());
        drop(R.write().unwrap());
        drop((R.read().unwrap(), R.read().unwrap()));
        drop(R.write().unwrap());
        unsafe { R.destroy(); }
    }

    #[test]
    fn frob() {
        static R: StaticRWLock = RWLOCK_INIT;
        static N: uint = 10;
        static M: uint = 1000;

        let (tx, rx) = channel::<()>();
        for _ in range(0, N) {
            let tx = tx.clone();
            spawn(move|| {
                let mut rng = rand::thread_rng();
                for _ in range(0, M) {
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
        let _ = rx.recv_opt();
        unsafe { R.destroy(); }
    }

    #[test]
    fn test_rw_arc_poison_wr() {
        let arc = Arc::new(RWLock::new(1i));
        let arc2 = arc.clone();
        let _: Result<uint, _> = Thread::spawn(move|| {
            let _lock = arc2.write().unwrap();
            panic!();
        }).join();
        assert!(arc.read().is_err());
    }

    #[test]
    fn test_rw_arc_poison_ww() {
        let arc = Arc::new(RWLock::new(1i));
        let arc2 = arc.clone();
        let _: Result<uint, _> = Thread::spawn(move|| {
            let _lock = arc2.write().unwrap();
            panic!();
        }).join();
        assert!(arc.write().is_err());
    }

    #[test]
    fn test_rw_arc_no_poison_rr() {
        let arc = Arc::new(RWLock::new(1i));
        let arc2 = arc.clone();
        let _: Result<uint, _> = Thread::spawn(move|| {
            let _lock = arc2.read().unwrap();
            panic!();
        }).join();
        let lock = arc.read().unwrap();
        assert_eq!(*lock, 1);
    }
    #[test]
    fn test_rw_arc_no_poison_rw() {
        let arc = Arc::new(RWLock::new(1i));
        let arc2 = arc.clone();
        let _: Result<uint, _> = Thread::spawn(move|| {
            let _lock = arc2.read().unwrap();
            panic!()
        }).join();
        let lock = arc.write().unwrap();
        assert_eq!(*lock, 1);
    }

    #[test]
    fn test_rw_arc() {
        let arc = Arc::new(RWLock::new(0i));
        let arc2 = arc.clone();
        let (tx, rx) = channel();

        Thread::spawn(move|| {
            let mut lock = arc2.write().unwrap();
            for _ in range(0u, 10) {
                let tmp = *lock;
                *lock = -1;
                Thread::yield_now();
                *lock = tmp + 1;
            }
            tx.send(());
        }).detach();

        // Readers try to catch the writer in the act
        let mut children = Vec::new();
        for _ in range(0u, 5) {
            let arc3 = arc.clone();
            children.push(Thread::spawn(move|| {
                let lock = arc3.read().unwrap();
                assert!(*lock >= 0);
            }));
        }

        // Wait for children to pass their asserts
        for r in children.into_iter() {
            assert!(r.join().is_ok());
        }

        // Wait for writer to finish
        rx.recv();
        let lock = arc.read().unwrap();
        assert_eq!(*lock, 10);
    }

    #[test]
    fn test_rw_arc_access_in_unwind() {
        let arc = Arc::new(RWLock::new(1i));
        let arc2 = arc.clone();
        let _ = Thread::spawn(move|| -> () {
            struct Unwinder {
                i: Arc<RWLock<int>>,
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
