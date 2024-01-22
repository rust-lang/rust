use crate::sync::atomic::{AtomicIsize, Ordering::Acquire};
use crate::thread::yield_now;

pub struct RwLock {
    /// The "mode" value indicates how many threads are waiting on this
    /// Mutex. Possible values are:
    ///    -1: The lock is locked for writing
    ///     0: The lock is unlocked
    ///   >=1: The lock is locked for reading
    ///
    /// This currently spins waiting for the lock to be freed. An
    /// optimization would be to involve the ticktimer server to
    /// coordinate unlocks.
    mode: AtomicIsize,
}

const RWLOCK_WRITING: isize = -1;
const RWLOCK_FREE: isize = 0;

unsafe impl Send for RwLock {}
unsafe impl Sync for RwLock {}

impl RwLock {
    #[inline]
    #[rustc_const_stable(feature = "const_locks", since = "1.63.0")]
    pub const fn new() -> RwLock {
        RwLock { mode: AtomicIsize::new(RWLOCK_FREE) }
    }

    #[inline]
    pub unsafe fn read(&self) {
        while !unsafe { self.try_read() } {
            yield_now();
        }
    }

    #[inline]
    pub unsafe fn try_read(&self) -> bool {
        self.mode
            .fetch_update(
                Acquire,
                Acquire,
                |v| if v == RWLOCK_WRITING { None } else { Some(v + 1) },
            )
            .is_ok()
    }

    #[inline]
    pub unsafe fn write(&self) {
        while !unsafe { self.try_write() } {
            yield_now();
        }
    }

    #[inline]
    pub unsafe fn try_write(&self) -> bool {
        self.mode.compare_exchange(RWLOCK_FREE, RWLOCK_WRITING, Acquire, Acquire).is_ok()
    }

    #[inline]
    pub unsafe fn read_unlock(&self) {
        let previous = self.mode.fetch_sub(1, Acquire);
        assert!(previous != RWLOCK_FREE);
        assert!(previous != RWLOCK_WRITING);
    }

    #[inline]
    pub unsafe fn write_unlock(&self) {
        assert_eq!(
            self.mode.compare_exchange(RWLOCK_WRITING, RWLOCK_FREE, Acquire, Acquire),
            Ok(RWLOCK_WRITING)
        );
    }
}
