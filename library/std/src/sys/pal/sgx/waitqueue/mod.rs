//! A simple queue implementation for synchronization primitives.
//!
//! This queue is used to implement condition variable and mutexes.
//!
//! Users of this API are expected to use the `WaitVariable<T>` type. Since
//! that type is not `Sync`, it needs to be protected by e.g., a `SpinMutex` to
//! allow shared access.
//!
//! Since userspace may send spurious wake-ups, the wakeup event state is
//! recorded in the enclave. The wakeup event state is protected by a spinlock.
//! The queue and associated wait state are stored in a `WaitVariable`.

#[cfg(test)]
mod tests;

mod spin_mutex;
mod unsafe_list;

use crate::num::NonZero;
use crate::ops::{Deref, DerefMut};
use crate::panic::{self, AssertUnwindSafe};
use crate::time::Duration;

use super::abi::thread;
use super::abi::usercalls;
use fortanix_sgx_abi::{Tcs, EV_UNPARK, WAIT_INDEFINITE};

pub use self::spin_mutex::{try_lock_or_false, SpinMutex, SpinMutexGuard};
use self::unsafe_list::{UnsafeList, UnsafeListEntry};

/// An queue entry in a `WaitQueue`.
struct WaitEntry {
    /// TCS address of the thread that is waiting
    tcs: Tcs,
    /// Whether this thread has been notified to be awoken
    wake: bool,
}

/// Data stored with a `WaitQueue` alongside it. This ensures accesses to the
/// queue and the data are synchronized, since the type itself is not `Sync`.
///
/// Consumers of this API should use a synchronization primitive for shared
/// access, such as `SpinMutex`.
#[derive(Default)]
pub struct WaitVariable<T> {
    queue: WaitQueue,
    lock: T,
}

impl<T> WaitVariable<T> {
    pub const fn new(var: T) -> Self {
        WaitVariable { queue: WaitQueue::new(), lock: var }
    }

    pub fn queue_empty(&self) -> bool {
        self.queue.is_empty()
    }

    pub fn lock_var(&self) -> &T {
        &self.lock
    }

    pub fn lock_var_mut(&mut self) -> &mut T {
        &mut self.lock
    }
}

#[derive(Copy, Clone)]
pub enum NotifiedTcs {
    Single(Tcs),
    All { count: NonZero<usize> },
}

/// An RAII guard that will notify a set of target threads as well as unlock
/// a mutex on drop.
pub struct WaitGuard<'a, T: 'a> {
    mutex_guard: Option<SpinMutexGuard<'a, WaitVariable<T>>>,
    notified_tcs: NotifiedTcs,
}

/// A queue of threads that are waiting on some synchronization primitive.
///
/// `UnsafeList` entries are allocated on the waiting thread's stack. This
/// avoids any global locking that might happen in the heap allocator. This is
/// safe because the waiting thread will not return from that stack frame until
/// after it is notified. The notifying thread ensures to clean up any
/// references to the list entries before sending the wakeup event.
pub struct WaitQueue {
    // We use an inner Mutex here to protect the data in the face of spurious
    // wakeups.
    inner: UnsafeList<SpinMutex<WaitEntry>>,
}
unsafe impl Send for WaitQueue {}

impl Default for WaitQueue {
    fn default() -> Self {
        Self::new()
    }
}

impl<'a, T> WaitGuard<'a, T> {
    /// Returns which TCSes will be notified when this guard drops.
    pub fn notified_tcs(&self) -> NotifiedTcs {
        self.notified_tcs
    }

    /// Drop this `WaitGuard`, after dropping another `guard`.
    pub fn drop_after<U>(self, guard: U) {
        drop(guard);
        drop(self);
    }
}

impl<'a, T> Deref for WaitGuard<'a, T> {
    type Target = SpinMutexGuard<'a, WaitVariable<T>>;

    fn deref(&self) -> &Self::Target {
        self.mutex_guard.as_ref().unwrap()
    }
}

impl<'a, T> DerefMut for WaitGuard<'a, T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.mutex_guard.as_mut().unwrap()
    }
}

impl<'a, T> Drop for WaitGuard<'a, T> {
    fn drop(&mut self) {
        drop(self.mutex_guard.take());
        let target_tcs = match self.notified_tcs {
            NotifiedTcs::Single(tcs) => Some(tcs),
            NotifiedTcs::All { .. } => None,
        };
        rtunwrap!(Ok, usercalls::send(EV_UNPARK, target_tcs));
    }
}

impl WaitQueue {
    pub const fn new() -> Self {
        WaitQueue { inner: UnsafeList::new() }
    }

    pub fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    /// Adds the calling thread to the `WaitVariable`'s wait queue, then wait
    /// until a wakeup event.
    ///
    /// This function does not return until this thread has been awoken. When `before_wait` panics,
    /// this function will abort.
    pub fn wait<T, F: FnOnce()>(mut guard: SpinMutexGuard<'_, WaitVariable<T>>, before_wait: F) {
        // very unsafe: check requirements of UnsafeList::push
        unsafe {
            let mut entry = UnsafeListEntry::new(SpinMutex::new(WaitEntry {
                tcs: thread::current(),
                wake: false,
            }));
            let entry = guard.queue.inner.push(&mut entry);
            drop(guard);
            if let Err(_e) = panic::catch_unwind(AssertUnwindSafe(|| before_wait())) {
                rtabort!("Panic before wait on wakeup event")
            }
            while !entry.lock().wake {
                // `entry.wake` is only set in `notify_one` and `notify_all` functions. Both ensure
                // the entry is removed from the queue _before_ setting this bool. There are no
                // other references to `entry`.
                // don't panic, this would invalidate `entry` during unwinding
                let eventset = rtunwrap!(Ok, usercalls::wait(EV_UNPARK, WAIT_INDEFINITE));
                rtassert!(eventset & EV_UNPARK == EV_UNPARK);
            }
        }
    }

    /// Adds the calling thread to the `WaitVariable`'s wait queue, then wait
    /// until a wakeup event or timeout. If event was observed, returns true.
    /// If not, it will remove the calling thread from the wait queue.
    /// When `before_wait` panics, this function will abort.
    pub fn wait_timeout<T, F: FnOnce()>(
        lock: &SpinMutex<WaitVariable<T>>,
        timeout: Duration,
        before_wait: F,
    ) -> bool {
        // very unsafe: check requirements of UnsafeList::push
        unsafe {
            let mut entry = UnsafeListEntry::new(SpinMutex::new(WaitEntry {
                tcs: thread::current(),
                wake: false,
            }));
            let entry_lock = lock.lock().queue.inner.push(&mut entry);
            if let Err(_e) = panic::catch_unwind(AssertUnwindSafe(|| before_wait())) {
                rtabort!("Panic before wait on wakeup event or timeout")
            }
            usercalls::wait_timeout(EV_UNPARK, timeout, || entry_lock.lock().wake);
            // acquire the wait queue's lock first to avoid deadlock
            // and ensure no other function can simultaneously access the list
            // (e.g., `notify_one` or `notify_all`)
            let mut guard = lock.lock();
            let success = entry_lock.lock().wake;
            if !success {
                // nobody is waking us up, so remove our entry from the wait queue.
                guard.queue.inner.remove(&mut entry);
            }
            success
        }
    }

    /// Either find the next waiter on the wait queue, or return the mutex
    /// guard unchanged.
    ///
    /// If a waiter is found, a `WaitGuard` is returned which will notify the
    /// waiter when it is dropped.
    pub fn notify_one<T>(
        mut guard: SpinMutexGuard<'_, WaitVariable<T>>,
    ) -> Result<WaitGuard<'_, T>, SpinMutexGuard<'_, WaitVariable<T>>> {
        // SAFETY: lifetime of the pop() return value is limited to the map
        // closure (The closure return value is 'static). The underlying
        // stack frame won't be freed until after the lock on the queue is released
        // (i.e., `guard` is dropped).
        unsafe {
            let tcs = guard.queue.inner.pop().map(|entry| -> Tcs {
                let mut entry_guard = entry.lock();
                entry_guard.wake = true;
                entry_guard.tcs
            });

            if let Some(tcs) = tcs {
                Ok(WaitGuard { mutex_guard: Some(guard), notified_tcs: NotifiedTcs::Single(tcs) })
            } else {
                Err(guard)
            }
        }
    }

    /// Either find any and all waiters on the wait queue, or return the mutex
    /// guard unchanged.
    ///
    /// If at least one waiter is found, a `WaitGuard` is returned which will
    /// notify all waiters when it is dropped.
    pub fn notify_all<T>(
        mut guard: SpinMutexGuard<'_, WaitVariable<T>>,
    ) -> Result<WaitGuard<'_, T>, SpinMutexGuard<'_, WaitVariable<T>>> {
        // SAFETY: lifetime of the pop() return values are limited to the
        // while loop body. The underlying stack frames won't be freed until
        // after the lock on the queue is released (i.e., `guard` is dropped).
        unsafe {
            let mut count = 0;
            while let Some(entry) = guard.queue.inner.pop() {
                count += 1;
                let mut entry_guard = entry.lock();
                entry_guard.wake = true;
            }

            if let Some(count) = NonZero::new(count) {
                Ok(WaitGuard { mutex_guard: Some(guard), notified_tcs: NotifiedTcs::All { count } })
            } else {
                Err(guard)
            }
        }
    }
}
