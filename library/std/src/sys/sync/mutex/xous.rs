use crate::os::xous::ffi::{blocking_scalar, do_yield};
use crate::os::xous::services::{TicktimerScalar, ticktimer_server};
use crate::sync::atomic::Ordering::{Acquire, Relaxed, Release};
use crate::sync::atomic::{Atomic, AtomicBool, AtomicUsize};

pub struct Mutex {
    /// The "locked" value indicates how many threads are waiting on this
    /// Mutex. Possible values are:
    ///     0: The lock is unlocked
    ///     1: The lock is locked and uncontended
    ///   >=2: The lock is locked and contended
    ///
    /// A lock is "contended" when there is more than one thread waiting
    /// for a lock, or it is locked for long periods of time. Rather than
    /// spinning, these locks send a Message to the ticktimer server
    /// requesting that they be woken up when a lock is unlocked.
    locked: Atomic<usize>,

    /// Whether this Mutex ever was contended, and therefore made a trip
    /// to the ticktimer server. If this was never set, then we were never
    /// on the slow path and can skip deregistering the mutex.
    contended: Atomic<bool>,
}

impl Mutex {
    #[inline]
    pub const fn new() -> Mutex {
        Mutex { locked: AtomicUsize::new(0), contended: AtomicBool::new(false) }
    }

    fn index(&self) -> usize {
        core::ptr::from_ref(self).addr()
    }

    #[inline]
    pub unsafe fn lock(&self) {
        // Try multiple times to acquire the lock without resorting to the ticktimer
        // server. For locks that are held for a short amount of time, this will
        // result in the ticktimer server never getting invoked. The `locked` value
        // will be either 0 or 1.
        for _attempts in 0..3 {
            if unsafe { self.try_lock() } {
                return;
            }
            do_yield();
        }

        // Try one more time to lock. If the lock is released between the previous code and
        // here, then the inner `locked` value will be 1 at the end of this. If it was not
        // locked, then the value will be more than 1, for example if there are multiple other
        // threads waiting on this lock.
        if unsafe { self.try_lock_or_poison() } {
            return;
        }

        // When this mutex is dropped, we will need to deregister it with the server.
        self.contended.store(true, Relaxed);

        // The lock is now "contended". When the lock is released, a Message will get sent to the
        // ticktimer server to wake it up. Note that this may already have happened, so the actual
        // value of `lock` may be anything (0, 1, 2, ...).
        blocking_scalar(
            ticktimer_server(),
            crate::os::xous::services::TicktimerScalar::LockMutex(self.index()).into(),
        )
        .expect("failure to send LockMutex command");
    }

    #[inline]
    pub unsafe fn unlock(&self) {
        let prev = self.locked.fetch_sub(1, Release);

        // If the previous value was 1, then this was a "fast path" unlock, so no
        // need to involve the Ticktimer server
        if prev == 1 {
            return;
        }

        // If it was 0, then something has gone seriously wrong and the counter
        // has just wrapped around.
        if prev == 0 {
            panic!("mutex lock count underflowed");
        }

        // Unblock one thread that is waiting on this message.
        blocking_scalar(ticktimer_server(), TicktimerScalar::UnlockMutex(self.index()).into())
            .expect("failure to send UnlockMutex command");
    }

    #[inline]
    pub unsafe fn try_lock(&self) -> bool {
        self.locked.compare_exchange(0, 1, Acquire, Relaxed).is_ok()
    }

    #[inline]
    pub unsafe fn try_lock_or_poison(&self) -> bool {
        self.locked.fetch_add(1, Acquire) == 0
    }
}

impl Drop for Mutex {
    fn drop(&mut self) {
        // If there was Mutex contention, then we involved the ticktimer. Free
        // the resources associated with this Mutex as it is deallocated.
        if self.contended.load(Relaxed) {
            blocking_scalar(ticktimer_server(), TicktimerScalar::FreeMutex(self.index()).into())
                .ok();
        }
    }
}
