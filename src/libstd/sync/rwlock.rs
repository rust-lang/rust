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

use kinds::marker;
use cell::UnsafeCell;
use sys_common::rwlock as sys;
use sync::poison;

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
/// RWLocks, like Mutexes, will become poisoned on panics. Note, however, that
/// an RWLock may only be poisoned if a panic occurs while it is locked
/// exclusively (write mode). If a panic occurs in any reader, then the lock
/// will not be poisoned.
///
/// # Example
///
/// ```
/// use std::sync::RWLock;
///
/// let lock = RWLock::new(5i);
///
/// // many reader locks can be held at once
/// {
///     let r1 = lock.read();
///     let r2 = lock.read();
///     assert_eq!(*r1, 5);
///     assert_eq!(*r2, 5);
/// } // read locks are dropped at this point
///
/// // only one write lock may be held, however
/// {
///     let mut w = lock.write();
///     *w += 1;
///     assert_eq!(*w, 6);
/// } // write lock is dropped here
/// ```
pub struct RWLock<T> {
    inner: Box<StaticRWLock>,
    data: UnsafeCell<T>,
}

/// Structure representing a staticaly allocated RWLock.
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
///     let _g = LOCK.read();
///     // ... shared read access
/// }
/// {
///     let _g = LOCK.write();
///     // ... exclusive write access
/// }
/// unsafe { LOCK.destroy() } // free all resources
/// ```
pub struct StaticRWLock {
    inner: sys::RWLock,
    poison: UnsafeCell<poison::Flag>,
}

/// Constant initialization for a statically-initialized rwlock.
pub const RWLOCK_INIT: StaticRWLock = StaticRWLock {
    inner: sys::RWLOCK_INIT,
    poison: UnsafeCell { value: poison::Flag { failed: false } },
};

/// RAII structure used to release the shared read access of a lock when
/// dropped.
#[must_use]
pub struct RWLockReadGuard<'a, T: 'a> {
    __lock: &'a RWLock<T>,
    __guard: StaticRWLockReadGuard,
}

/// RAII structure used to release the exclusive write access of a lock when
/// dropped.
#[must_use]
pub struct RWLockWriteGuard<'a, T: 'a> {
    __lock: &'a RWLock<T>,
    __guard: StaticRWLockWriteGuard,
}

/// RAII structure used to release the shared read access of a lock when
/// dropped.
#[must_use]
pub struct StaticRWLockReadGuard {
    lock: &'static sys::RWLock,
    marker: marker::NoSend,
}

/// RAII structure used to release the exclusive write access of a lock when
/// dropped.
#[must_use]
pub struct StaticRWLockWriteGuard {
    lock: &'static sys::RWLock,
    marker: marker::NoSend,
    poison: poison::Guard<'static>,
}

impl<T: Send + Sync> RWLock<T> {
    /// Creates a new instance of an RWLock which is unlocked and read to go.
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
    /// # Panics
    ///
    /// This function will panic if the RWLock is poisoned. An RWLock is
    /// poisoned whenever a writer panics while holding an exclusive lock. The
    /// panic will occur immediately after the lock has been acquired.
    #[inline]
    pub fn read(&self) -> RWLockReadGuard<T> {
        unsafe {
            let lock: &'static StaticRWLock = &*(&*self.inner as *const _);
            RWLockReadGuard::new(self, lock.read())
        }
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
    /// # Panics
    ///
    /// This function will panic if the RWLock is poisoned. An RWLock is
    /// poisoned whenever a writer panics while holding an exclusive lock. A
    /// panic will only occur if the lock is acquired.
    #[inline]
    pub fn try_read(&self) -> Option<RWLockReadGuard<T>> {
        unsafe {
            let lock: &'static StaticRWLock = &*(&*self.inner as *const _);
            lock.try_read().map(|guard| {
                RWLockReadGuard::new(self, guard)
            })
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
    /// # Panics
    ///
    /// This function will panic if the RWLock is poisoned. An RWLock is
    /// poisoned whenever a writer panics while holding an exclusive lock. The
    /// panic will occur when the lock is acquired.
    #[inline]
    pub fn write(&self) -> RWLockWriteGuard<T> {
        unsafe {
            let lock: &'static StaticRWLock = &*(&*self.inner as *const _);
            RWLockWriteGuard::new(self, lock.write())
        }
    }

    /// Attempt to lock this rwlock with exclusive write access.
    ///
    /// This function does not ever block, and it will return `None` if a call
    /// to `write` would otherwise block. If successful, an RAII guard is
    /// returned.
    ///
    /// # Panics
    ///
    /// This function will panic if the RWLock is poisoned. An RWLock is
    /// poisoned whenever a writer panics while holding an exclusive lock. A
    /// panic will only occur if the lock is acquired.
    #[inline]
    pub fn try_write(&self) -> Option<RWLockWriteGuard<T>> {
        unsafe {
            let lock: &'static StaticRWLock = &*(&*self.inner as *const _);
            lock.try_write().map(|guard| {
                RWLockWriteGuard::new(self, guard)
            })
        }
    }
}

#[unsafe_destructor]
impl<T> Drop for RWLock<T> {
    fn drop(&mut self) {
        unsafe { self.inner.inner.destroy() }
    }
}

impl StaticRWLock {
    /// Locks this rwlock with shared read access, blocking the current thread
    /// until it can be acquired.
    ///
    /// See `RWLock::read`.
    #[inline]
    pub fn read(&'static self) -> StaticRWLockReadGuard {
        unsafe { self.inner.read() }
        StaticRWLockReadGuard::new(self)
    }

    /// Attempt to acquire this lock with shared read access.
    ///
    /// See `RWLock::try_read`.
    #[inline]
    pub fn try_read(&'static self) -> Option<StaticRWLockReadGuard> {
        if unsafe { self.inner.try_read() } {
            Some(StaticRWLockReadGuard::new(self))
        } else {
            None
        }
    }

    /// Lock this rwlock with exclusive write access, blocking the current
    /// thread until it can be acquired.
    ///
    /// See `RWLock::write`.
    #[inline]
    pub fn write(&'static self) -> StaticRWLockWriteGuard {
        unsafe { self.inner.write() }
        StaticRWLockWriteGuard::new(self)
    }

    /// Attempt to lock this rwlock with exclusive write access.
    ///
    /// See `RWLock::try_write`.
    #[inline]
    pub fn try_write(&'static self) -> Option<StaticRWLockWriteGuard> {
        if unsafe { self.inner.try_write() } {
            Some(StaticRWLockWriteGuard::new(self))
        } else {
            None
        }
    }

    /// Deallocate all resources associated with this static lock.
    ///
    /// This method is unsafe to call as there is no guarantee that there are no
    /// active users of the lock, and this also doesn't prevent any future users
    /// of this lock. This method is required to be called to not leak memory on
    /// all platforms.
    pub unsafe fn destroy(&'static self) {
        self.inner.destroy()
    }
}

impl<'rwlock, T> RWLockReadGuard<'rwlock, T> {
    fn new(lock: &RWLock<T>, guard: StaticRWLockReadGuard)
           -> RWLockReadGuard<T> {
        RWLockReadGuard { __lock: lock, __guard: guard }
    }
}
impl<'rwlock, T> RWLockWriteGuard<'rwlock, T> {
    fn new(lock: &RWLock<T>, guard: StaticRWLockWriteGuard)
           -> RWLockWriteGuard<T> {
        RWLockWriteGuard { __lock: lock, __guard: guard }
    }
}

impl<'rwlock, T> Deref<T> for RWLockReadGuard<'rwlock, T> {
    fn deref(&self) -> &T { unsafe { &*self.__lock.data.get() } }
}
impl<'rwlock, T> Deref<T> for RWLockWriteGuard<'rwlock, T> {
    fn deref(&self) -> &T { unsafe { &*self.__lock.data.get() } }
}
impl<'rwlock, T> DerefMut<T> for RWLockWriteGuard<'rwlock, T> {
    fn deref_mut(&mut self) -> &mut T { unsafe { &mut *self.__lock.data.get() } }
}

impl StaticRWLockReadGuard {
    fn new(lock: &'static StaticRWLock) -> StaticRWLockReadGuard {
        let guard = StaticRWLockReadGuard {
            lock: &lock.inner,
            marker: marker::NoSend,
        };
        unsafe { (*lock.poison.get()).borrow().check("rwlock"); }
        return guard;
    }
}
impl StaticRWLockWriteGuard {
    fn new(lock: &'static StaticRWLock) -> StaticRWLockWriteGuard {
        unsafe {
            let guard = StaticRWLockWriteGuard {
                lock: &lock.inner,
                marker: marker::NoSend,
                poison: (*lock.poison.get()).borrow(),
            };
            guard.poison.check("rwlock");
            return guard;
        }
    }
}

#[unsafe_destructor]
impl Drop for StaticRWLockReadGuard {
    fn drop(&mut self) {
        unsafe { self.lock.read_unlock(); }
    }
}

#[unsafe_destructor]
impl Drop for StaticRWLockWriteGuard {
    fn drop(&mut self) {
        self.poison.done();
        unsafe { self.lock.write_unlock(); }
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
        drop(l.read());
        drop(l.write());
        drop((l.read(), l.read()));
        drop(l.write());
    }

    #[test]
    fn static_smoke() {
        static R: StaticRWLock = RWLOCK_INIT;
        drop(R.read());
        drop(R.write());
        drop((R.read(), R.read()));
        drop(R.write());
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
                let mut rng = rand::task_rng();
                for _ in range(0, M) {
                    if rng.gen_weighted_bool(N) {
                        drop(R.write());
                    } else {
                        drop(R.read());
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
    #[should_fail]
    fn test_rw_arc_poison_wr() {
        let arc = Arc::new(RWLock::new(1i));
        let arc2 = arc.clone();
        let _ = Thread::spawn(move|| {
            let lock = arc2.write();
            assert_eq!(*lock, 2);
        }).join();
        let lock = arc.read();
        assert_eq!(*lock, 1);
    }

    #[test]
    #[should_fail]
    fn test_rw_arc_poison_ww() {
        let arc = Arc::new(RWLock::new(1i));
        let arc2 = arc.clone();
        let _ = Thread::spawn(move|| {
            let lock = arc2.write();
            assert_eq!(*lock, 2);
        }).join();
        let lock = arc.write();
        assert_eq!(*lock, 1);
    }

    #[test]
    fn test_rw_arc_no_poison_rr() {
        let arc = Arc::new(RWLock::new(1i));
        let arc2 = arc.clone();
        let _ = Thread::spawn(move|| {
            let lock = arc2.read();
            assert_eq!(*lock, 2);
        }).join();
        let lock = arc.read();
        assert_eq!(*lock, 1);
    }
    #[test]
    fn test_rw_arc_no_poison_rw() {
        let arc = Arc::new(RWLock::new(1i));
        let arc2 = arc.clone();
        let _ = Thread::spawn(move|| {
            let lock = arc2.read();
            assert_eq!(*lock, 2);
        }).join();
        let lock = arc.write();
        assert_eq!(*lock, 1);
    }

    #[test]
    fn test_rw_arc() {
        let arc = Arc::new(RWLock::new(0i));
        let arc2 = arc.clone();
        let (tx, rx) = channel();

        Thread::spawn(move|| {
            let mut lock = arc2.write();
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
                let lock = arc3.read();
                assert!(*lock >= 0);
            }));
        }

        // Wait for children to pass their asserts
        for r in children.into_iter() {
            assert!(r.join().is_ok());
        }

        // Wait for writer to finish
        rx.recv();
        let lock = arc.read();
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
                    let mut lock = self.i.write();
                    *lock += 1;
                }
            }
            let _u = Unwinder { i: arc2 };
            panic!();
        }).join();
        let lock = arc.read();
        assert_eq!(*lock, 2);
    }
}
