// Copyright 2012-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Wrappers for safe, shared, mutable memory between tasks
//!
//! The wrappers in this module build on the primitives from `sync::raw` to
//! provide safe interfaces around using the primitive locks. These primitives
//! implement a technique called "poisoning" where when a task failed with a
//! held lock, all future attempts to use the lock will fail.
//!
//! For example, if two tasks are contending on a mutex and one of them fails
//! after grabbing the lock, the second task will immediately fail because the
//! lock is now poisoned.

use std::task;
use std::ty::Unsafe;

use raw;

/****************************************************************************
 * Poisoning helpers
 ****************************************************************************/

struct PoisonOnFail<'a> {
    flag: &'a mut bool,
    failed: bool,
}

impl<'a> PoisonOnFail<'a> {
    fn check(flag: bool, name: &str) {
        if flag {
            fail!("Poisoned {} - another task failed inside!", name);
        }
    }

    fn new<'a>(flag: &'a mut bool, name: &str) -> PoisonOnFail<'a> {
        PoisonOnFail::check(*flag, name);
        PoisonOnFail {
            flag: flag,
            failed: task::failing()
        }
    }
}

#[unsafe_destructor]
impl<'a> Drop for PoisonOnFail<'a> {
    fn drop(&mut self) {
        if !self.failed && task::failing() {
            *self.flag = true;
        }
    }
}

/****************************************************************************
 * Condvar
 ****************************************************************************/

enum Inner<'a> {
    InnerMutex(raw::MutexGuard<'a>),
    InnerRWLock(raw::RWLockWriteGuard<'a>),
}

impl<'b> Inner<'b> {
    fn cond<'a>(&'a self) -> &'a raw::Condvar<'b> {
        match *self {
            InnerMutex(ref m) => &m.cond,
            InnerRWLock(ref m) => &m.cond,
        }
    }
}

/// A condition variable, a mechanism for unlock-and-descheduling and
/// signaling, for use with the lock types.
pub struct Condvar<'a> {
    name: &'static str,
    // n.b. Inner must be after PoisonOnFail because we must set the poison flag
    //      *inside* the mutex, and struct fields are destroyed top-to-bottom
    //      (destroy the lock guard last).
    poison: PoisonOnFail<'a>,
    inner: Inner<'a>,
}

impl<'a> Condvar<'a> {
    /// Atomically exit the associated lock and block until a signal is sent.
    ///
    /// wait() is equivalent to wait_on(0).
    ///
    /// # Failure
    ///
    /// A task which is killed while waiting on a condition variable will wake
    /// up, fail, and unlock the associated lock as it unwinds.
    #[inline]
    pub fn wait(&self) { self.wait_on(0) }

    /// Atomically exit the associated lock and block on a specified condvar
    /// until a signal is sent on that same condvar.
    ///
    /// The associated lock must have been initialised with an appropriate
    /// number of condvars. The condvar_id must be between 0 and num_condvars-1
    /// or else this call will fail.
    #[inline]
    pub fn wait_on(&self, condvar_id: uint) {
        assert!(!*self.poison.flag);
        self.inner.cond().wait_on(condvar_id);
        // This is why we need to wrap sync::condvar.
        PoisonOnFail::check(*self.poison.flag, self.name);
    }

    /// Wake up a blocked task. Returns false if there was no blocked task.
    #[inline]
    pub fn signal(&self) -> bool { self.signal_on(0) }

    /// Wake up a blocked task on a specified condvar (as
    /// sync::cond.signal_on). Returns false if there was no blocked task.
    #[inline]
    pub fn signal_on(&self, condvar_id: uint) -> bool {
        assert!(!*self.poison.flag);
        self.inner.cond().signal_on(condvar_id)
    }

    /// Wake up all blocked tasks. Returns the number of tasks woken.
    #[inline]
    pub fn broadcast(&self) -> uint { self.broadcast_on(0) }

    /// Wake up all blocked tasks on a specified condvar (as
    /// sync::cond.broadcast_on). Returns the number of tasks woken.
    #[inline]
    pub fn broadcast_on(&self, condvar_id: uint) -> uint {
        assert!(!*self.poison.flag);
        self.inner.cond().broadcast_on(condvar_id)
    }
}

/****************************************************************************
 * Mutex
 ****************************************************************************/

/// A wrapper type which provides synchronized access to the underlying data, of
/// type `T`. A mutex always provides exclusive access, and concurrent requests
/// will block while the mutex is already locked.
///
/// # Example
///
/// ```
/// use sync::{Mutex, Arc};
///
/// let mutex = Arc::new(Mutex::new(1));
/// let mutex2 = mutex.clone();
///
/// spawn(proc() {
///     let mut val = mutex2.lock();
///     *val += 1;
///     val.cond.signal();
/// });
///
/// let mut value = mutex.lock();
/// while *value != 2 {
///     value.cond.wait();
/// }
/// ```
pub struct Mutex<T> {
    lock: raw::Mutex,
    failed: Unsafe<bool>,
    data: Unsafe<T>,
}

/// An guard which is created by locking a mutex. Through this guard the
/// underlying data can be accessed.
pub struct MutexGuard<'a, T> {
    // FIXME #12808: strange name to try to avoid interfering with
    // field accesses of the contained type via Deref
    _data: &'a mut T,
    /// Inner condition variable connected to the locked mutex that this guard
    /// was created from. This can be used for atomic-unlock-and-deschedule.
    pub cond: Condvar<'a>,
}

impl<T: Send> Mutex<T> {
    /// Creates a new mutex to protect the user-supplied data.
    pub fn new(user_data: T) -> Mutex<T> {
        Mutex::new_with_condvars(user_data, 1)
    }

    /// Create a new mutex, with a specified number of associated condvars.
    ///
    /// This will allow calling wait_on/signal_on/broadcast_on with condvar IDs
    /// between 0 and num_condvars-1. (If num_condvars is 0, lock_cond will be
    /// allowed but any operations on the condvar will fail.)
    pub fn new_with_condvars(user_data: T, num_condvars: uint) -> Mutex<T> {
        Mutex {
            lock: raw::Mutex::new_with_condvars(num_condvars),
            failed: Unsafe::new(false),
            data: Unsafe::new(user_data),
        }
    }

    /// Access the underlying mutable data with mutual exclusion from other
    /// tasks. The returned value is an RAII guard which will unlock the mutex
    /// when dropped. All concurrent tasks attempting to lock the mutex will
    /// block while the returned value is still alive.
    ///
    /// # Failure
    ///
    /// Failing while inside the Mutex will unlock the Mutex while unwinding, so
    /// that other tasks won't block forever. It will also poison the Mutex:
    /// any tasks that subsequently try to access it (including those already
    /// blocked on the mutex) will also fail immediately.
    #[inline]
    pub fn lock<'a>(&'a self) -> MutexGuard<'a, T> {
        let guard = self.lock.lock();

        // These two accesses are safe because we're guranteed at this point
        // that we have exclusive access to this mutex. We are indeed able to
        // promote ourselves from &Mutex to `&mut T`
        let poison = unsafe { &mut *self.failed.get() };
        let data = unsafe { &mut *self.data.get() };

        MutexGuard {
            _data: data,
            cond: Condvar {
                name: "Mutex",
                poison: PoisonOnFail::new(poison, "Mutex"),
                inner: InnerMutex(guard),
            },
        }
    }
}

impl<'a, T: Send> Deref<T> for MutexGuard<'a, T> {
    fn deref<'a>(&'a self) -> &'a T { &*self._data }
}
impl<'a, T: Send> DerefMut<T> for MutexGuard<'a, T> {
    fn deref_mut<'a>(&'a mut self) -> &'a mut T { &mut *self._data }
}

/****************************************************************************
 * R/W lock protected lock
 ****************************************************************************/

/// A dual-mode reader-writer lock. The data can be accessed mutably or
/// immutably, and immutably-accessing tasks may run concurrently.
///
/// # Example
///
/// ```
/// use sync::{RWLock, Arc};
///
/// let lock1 = Arc::new(RWLock::new(1));
/// let lock2 = lock1.clone();
///
/// spawn(proc() {
///     let mut val = lock2.write();
///     *val = 3;
///     let val = val.downgrade();
///     println!("{}", *val);
/// });
///
/// let val = lock1.read();
/// println!("{}", *val);
/// ```
pub struct RWLock<T> {
    lock: raw::RWLock,
    failed: Unsafe<bool>,
    data: Unsafe<T>,
}

/// A guard which is created by locking an rwlock in write mode. Through this
/// guard the underlying data can be accessed.
pub struct RWLockWriteGuard<'a, T> {
    // FIXME #12808: strange name to try to avoid interfering with
    // field accesses of the contained type via Deref
    _data: &'a mut T,
    /// Inner condition variable that can be used to sleep on the write mode of
    /// this rwlock.
    pub cond: Condvar<'a>,
}

/// A guard which is created by locking an rwlock in read mode. Through this
/// guard the underlying data can be accessed.
pub struct RWLockReadGuard<'a, T> {
    // FIXME #12808: strange names to try to avoid interfering with
    // field accesses of the contained type via Deref
    _data: &'a T,
    _guard: raw::RWLockReadGuard<'a>,
}

impl<T: Send + Share> RWLock<T> {
    /// Create a reader/writer lock with the supplied data.
    pub fn new(user_data: T) -> RWLock<T> {
        RWLock::new_with_condvars(user_data, 1)
    }

    /// Create a reader/writer lock with the supplied data and a specified number
    /// of condvars (as sync::RWLock::new_with_condvars).
    pub fn new_with_condvars(user_data: T, num_condvars: uint) -> RWLock<T> {
        RWLock {
            lock: raw::RWLock::new_with_condvars(num_condvars),
            failed: Unsafe::new(false),
            data: Unsafe::new(user_data),
        }
    }

    /// Access the underlying data mutably. Locks the rwlock in write mode;
    /// other readers and writers will block.
    ///
    /// # Failure
    ///
    /// Failing while inside the lock will unlock the lock while unwinding, so
    /// that other tasks won't block forever. As Mutex.lock, it will also poison
    /// the lock, so subsequent readers and writers will both also fail.
    #[inline]
    pub fn write<'a>(&'a self) -> RWLockWriteGuard<'a, T> {
        let guard = self.lock.write();

        // These two accesses are safe because we're guranteed at this point
        // that we have exclusive access to this rwlock. We are indeed able to
        // promote ourselves from &RWLock to `&mut T`
        let poison = unsafe { &mut *self.failed.get() };
        let data = unsafe { &mut *self.data.get() };

        RWLockWriteGuard {
            _data: data,
            cond: Condvar {
                name: "RWLock",
                poison: PoisonOnFail::new(poison, "RWLock"),
                inner: InnerRWLock(guard),
            },
        }
    }

    /// Access the underlying data immutably. May run concurrently with other
    /// reading tasks.
    ///
    /// # Failure
    ///
    /// Failing will unlock the lock while unwinding. However, unlike all other
    /// access modes, this will not poison the lock.
    pub fn read<'a>(&'a self) -> RWLockReadGuard<'a, T> {
        let guard = self.lock.read();
        PoisonOnFail::check(unsafe { *self.failed.get() }, "RWLock");
        RWLockReadGuard {
            _guard: guard,
            _data: unsafe { &*self.data.get() },
        }
    }
}

impl<'a, T: Send + Share> RWLockWriteGuard<'a, T> {
    /// Consumes this write lock token, returning a new read lock token.
    ///
    /// This will allow pending readers to come into the lock.
    pub fn downgrade(self) -> RWLockReadGuard<'a, T> {
        let RWLockWriteGuard { _data, cond } = self;
        // convert the data to read-only explicitly
        let data = &*_data;
        let guard = match cond.inner {
            InnerMutex(..) => unreachable!(),
            InnerRWLock(guard) => guard.downgrade()
        };
        RWLockReadGuard { _guard: guard, _data: data }
    }
}

impl<'a, T: Send + Share> Deref<T> for RWLockReadGuard<'a, T> {
    fn deref<'a>(&'a self) -> &'a T { self._data }
}
impl<'a, T: Send + Share> Deref<T> for RWLockWriteGuard<'a, T> {
    fn deref<'a>(&'a self) -> &'a T { &*self._data }
}
impl<'a, T: Send + Share> DerefMut<T> for RWLockWriteGuard<'a, T> {
    fn deref_mut<'a>(&'a mut self) -> &'a mut T { &mut *self._data }
}

/****************************************************************************
 * Barrier
 ****************************************************************************/

/// A barrier enables multiple tasks to synchronize the beginning
/// of some computation.
///
/// ```rust
/// use sync::{Arc, Barrier};
///
/// let barrier = Arc::new(Barrier::new(10));
/// for _ in range(0, 10) {
///     let c = barrier.clone();
///     // The same messages will be printed together.
///     // You will NOT see any interleaving.
///     spawn(proc() {
///         println!("before wait");
///         c.wait();
///         println!("after wait");
///     });
/// }
/// ```
pub struct Barrier {
    lock: Mutex<BarrierState>,
    num_tasks: uint,
}

// The inner state of a double barrier
struct BarrierState {
    count: uint,
    generation_id: uint,
}

impl Barrier {
    /// Create a new barrier that can block a given number of tasks.
    pub fn new(num_tasks: uint) -> Barrier {
        Barrier {
            lock: Mutex::new(BarrierState {
                count: 0,
                generation_id: 0,
            }),
            num_tasks: num_tasks,
        }
    }

    /// Block the current task until a certain number of tasks is waiting.
    pub fn wait(&self) {
        let mut lock = self.lock.lock();
        let local_gen = lock.generation_id;
        lock.count += 1;
        if lock.count < self.num_tasks {
            // We need a while loop to guard against spurious wakeups.
            // http://en.wikipedia.org/wiki/Spurious_wakeup
            while local_gen == lock.generation_id &&
                  lock.count < self.num_tasks {
                lock.cond.wait();
            }
        } else {
            lock.count = 0;
            lock.generation_id += 1;
            lock.cond.broadcast();
        }
    }
}

/****************************************************************************
 * Tests
 ****************************************************************************/

#[cfg(test)]
mod tests {
    use std::comm::Empty;
    use std::task;
    use std::task::TaskBuilder;

    use Arc;
    use super::{Mutex, Barrier, RWLock};

    #[test]
    fn test_mutex_arc_condvar() {
        let arc = Arc::new(Mutex::new(false));
        let arc2 = arc.clone();
        let (tx, rx) = channel();
        task::spawn(proc() {
            // wait until parent gets in
            rx.recv();
            let mut lock = arc2.lock();
            *lock = true;
            lock.cond.signal();
        });

        let lock = arc.lock();
        tx.send(());
        assert!(!*lock);
        while !*lock {
            lock.cond.wait();
        }
    }

    #[test] #[should_fail]
    fn test_arc_condvar_poison() {
        let arc = Arc::new(Mutex::new(1));
        let arc2 = arc.clone();
        let (tx, rx) = channel();

        spawn(proc() {
            rx.recv();
            let lock = arc2.lock();
            lock.cond.signal();
            // Parent should fail when it wakes up.
            fail!();
        });

        let lock = arc.lock();
        tx.send(());
        while *lock == 1 {
            lock.cond.wait();
        }
    }

    #[test] #[should_fail]
    fn test_mutex_arc_poison() {
        let arc = Arc::new(Mutex::new(1));
        let arc2 = arc.clone();
        let _ = task::try(proc() {
            let lock = arc2.lock();
            assert_eq!(*lock, 2);
        });
        let lock = arc.lock();
        assert_eq!(*lock, 1);
    }

    #[test]
    fn test_mutex_arc_nested() {
        // Tests nested mutexes and access
        // to underlying data.
        let arc = Arc::new(Mutex::new(1));
        let arc2 = Arc::new(Mutex::new(arc));
        task::spawn(proc() {
            let lock = arc2.lock();
            let lock2 = lock.deref().lock();
            assert_eq!(*lock2, 1);
        });
    }

    #[test]
    fn test_mutex_arc_access_in_unwind() {
        let arc = Arc::new(Mutex::new(1i));
        let arc2 = arc.clone();
        let _ = task::try::<()>(proc() {
            struct Unwinder {
                i: Arc<Mutex<int>>,
            }
            impl Drop for Unwinder {
                fn drop(&mut self) {
                    let mut lock = self.i.lock();
                    *lock += 1;
                }
            }
            let _u = Unwinder { i: arc2 };
            fail!();
        });
        let lock = arc.lock();
        assert_eq!(*lock, 2);
    }

    #[test] #[should_fail]
    fn test_rw_arc_poison_wr() {
        let arc = Arc::new(RWLock::new(1));
        let arc2 = arc.clone();
        let _ = task::try(proc() {
            let lock = arc2.write();
            assert_eq!(*lock, 2);
        });
        let lock = arc.read();
        assert_eq!(*lock, 1);
    }
    #[test] #[should_fail]
    fn test_rw_arc_poison_ww() {
        let arc = Arc::new(RWLock::new(1));
        let arc2 = arc.clone();
        let _ = task::try(proc() {
            let lock = arc2.write();
            assert_eq!(*lock, 2);
        });
        let lock = arc.write();
        assert_eq!(*lock, 1);
    }
    #[test]
    fn test_rw_arc_no_poison_rr() {
        let arc = Arc::new(RWLock::new(1));
        let arc2 = arc.clone();
        let _ = task::try(proc() {
            let lock = arc2.read();
            assert_eq!(*lock, 2);
        });
        let lock = arc.read();
        assert_eq!(*lock, 1);
    }
    #[test]
    fn test_rw_arc_no_poison_rw() {
        let arc = Arc::new(RWLock::new(1));
        let arc2 = arc.clone();
        let _ = task::try(proc() {
            let lock = arc2.read();
            assert_eq!(*lock, 2);
        });
        let lock = arc.write();
        assert_eq!(*lock, 1);
    }
    #[test]
    fn test_rw_arc_no_poison_dr() {
        let arc = Arc::new(RWLock::new(1));
        let arc2 = arc.clone();
        let _ = task::try(proc() {
            let lock = arc2.write().downgrade();
            assert_eq!(*lock, 2);
        });
        let lock = arc.write();
        assert_eq!(*lock, 1);
    }

    #[test]
    fn test_rw_arc() {
        let arc = Arc::new(RWLock::new(0));
        let arc2 = arc.clone();
        let (tx, rx) = channel();

        task::spawn(proc() {
            let mut lock = arc2.write();
            for _ in range(0, 10) {
                let tmp = *lock;
                *lock = -1;
                task::deschedule();
                *lock = tmp + 1;
            }
            tx.send(());
        });

        // Readers try to catch the writer in the act
        let mut children = Vec::new();
        for _ in range(0, 5) {
            let arc3 = arc.clone();
            let mut builder = TaskBuilder::new();
            children.push(builder.future_result());
            builder.spawn(proc() {
                let lock = arc3.read();
                assert!(*lock >= 0);
            });
        }

        // Wait for children to pass their asserts
        for r in children.mut_iter() {
            assert!(r.recv().is_ok());
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
        let _ = task::try::<()>(proc() {
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
            fail!();
        });
        let lock = arc.read();
        assert_eq!(*lock, 2);
    }

    #[test]
    fn test_rw_downgrade() {
        // (1) A downgrader gets in write mode and does cond.wait.
        // (2) A writer gets in write mode, sets state to 42, and does signal.
        // (3) Downgrader wakes, sets state to 31337.
        // (4) tells writer and all other readers to contend as it downgrades.
        // (5) Writer attempts to set state back to 42, while downgraded task
        //     and all reader tasks assert that it's 31337.
        let arc = Arc::new(RWLock::new(0));

        // Reader tasks
        let mut reader_convos = Vec::new();
        for _ in range(0, 10) {
            let ((tx1, rx1), (tx2, rx2)) = (channel(), channel());
            reader_convos.push((tx1, rx2));
            let arcn = arc.clone();
            task::spawn(proc() {
                rx1.recv(); // wait for downgrader to give go-ahead
                let lock = arcn.read();
                assert_eq!(*lock, 31337);
                tx2.send(());
            });
        }

        // Writer task
        let arc2 = arc.clone();
        let ((tx1, rx1), (tx2, rx2)) = (channel(), channel());
        task::spawn(proc() {
            rx1.recv();
            {
                let mut lock = arc2.write();
                assert_eq!(*lock, 0);
                *lock = 42;
                lock.cond.signal();
            }
            rx1.recv();
            {
                let mut lock = arc2.write();
                // This shouldn't happen until after the downgrade read
                // section, and all other readers, finish.
                assert_eq!(*lock, 31337);
                *lock = 42;
            }
            tx2.send(());
        });

        // Downgrader (us)
        let mut lock = arc.write();
        tx1.send(()); // send to another writer who will wake us up
        while *lock == 0 {
            lock.cond.wait();
        }
        assert_eq!(*lock, 42);
        *lock = 31337;
        // send to other readers
        for &(ref mut rc, _) in reader_convos.mut_iter() {
            rc.send(())
        }
        let lock = lock.downgrade();
        // complete handshake with other readers
        for &(_, ref mut rp) in reader_convos.mut_iter() {
            rp.recv()
        }
        tx1.send(()); // tell writer to try again
        assert_eq!(*lock, 31337);
        drop(lock);

        rx2.recv(); // complete handshake with writer
    }

    #[cfg(test)]
    fn test_rw_write_cond_downgrade_read_race_helper() {
        // Tests that when a downgrader hands off the "reader cloud" lock
        // because of a contending reader, a writer can't race to get it
        // instead, which would result in readers_and_writers. This tests
        // the raw module rather than this one, but it's here because an
        // rwarc gives us extra shared state to help check for the race.
        let x = Arc::new(RWLock::new(true));
        let (tx, rx) = channel();

        // writer task
        let xw = x.clone();
        task::spawn(proc() {
            let mut lock = xw.write();
            tx.send(()); // tell downgrader it's ok to go
            lock.cond.wait();
            // The core of the test is here: the condvar reacquire path
            // must involve order_lock, so that it cannot race with a reader
            // trying to receive the "reader cloud lock hand-off".
            *lock = false;
        });

        rx.recv(); // wait for writer to get in

        let lock = x.write();
        assert!(*lock);
        // make writer contend in the cond-reacquire path
        lock.cond.signal();
        // make a reader task to trigger the "reader cloud lock" handoff
        let xr = x.clone();
        let (tx, rx) = channel();
        task::spawn(proc() {
            tx.send(());
            drop(xr.read());
        });
        rx.recv(); // wait for reader task to exist

        let lock = lock.downgrade();
        // if writer mistakenly got in, make sure it mutates state
        // before we assert on it
        for _ in range(0, 5) { task::deschedule(); }
        // make sure writer didn't get in.
        assert!(*lock);
    }
    #[test]
    fn test_rw_write_cond_downgrade_read_race() {
        // Ideally the above test case would have deschedule statements in it
        // that helped to expose the race nearly 100% of the time... but adding
        // deschedules in the intuitively-right locations made it even less
        // likely, and I wasn't sure why :( . This is a mediocre "next best"
        // option.
        for _ in range(0, 8) {
            test_rw_write_cond_downgrade_read_race_helper();
        }
    }

    /************************************************************************
     * Barrier tests
     ************************************************************************/
    #[test]
    fn test_barrier() {
        let barrier = Arc::new(Barrier::new(10));
        let (tx, rx) = channel();

        for _ in range(0, 9) {
            let c = barrier.clone();
            let tx = tx.clone();
            spawn(proc() {
                c.wait();
                tx.send(true);
            });
        }

        // At this point, all spawned tasks should be blocked,
        // so we shouldn't get anything from the port
        assert!(match rx.try_recv() {
            Err(Empty) => true,
            _ => false,
        });

        barrier.wait();
        // Now, the barrier is cleared and we should get data.
        for _ in range(0, 9) {
            rx.recv();
        }
    }
}
