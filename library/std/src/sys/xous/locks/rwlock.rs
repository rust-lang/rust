use crate::os::xous::ffi::do_yield;
use crate::sync::atomic::{AtomicIsize, Ordering::SeqCst};

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

unsafe impl Send for RwLock {}
unsafe impl Sync for RwLock {}

impl RwLock {
    #[inline]
    #[rustc_const_stable(feature = "const_locks", since = "1.63.0")]
    pub const fn new() -> RwLock {
        RwLock { mode: AtomicIsize::new(0) }
    }

    #[inline]
    pub unsafe fn read(&self) {
        while !unsafe { self.try_read() } {
            do_yield();
        }
    }

    #[inline]
    pub unsafe fn try_read(&self) -> bool {
        // Non-atomically determine the current value.
        let current = self.mode.load(SeqCst);

        // If it's currently locked for writing, then we cannot read.
        if current < 0 {
            return false;
        }

        // Attempt to lock. If the `current` value has changed, then this
        // operation will fail and we will not obtain the lock even if we
        // could potentially keep it.
        let new = current + 1;
        self.mode.compare_exchange(current, new, SeqCst, SeqCst).is_ok()
    }

    #[inline]
    pub unsafe fn write(&self) {
        while !unsafe { self.try_write() } {
            do_yield();
        }
    }

    #[inline]
    pub unsafe fn try_write(&self) -> bool {
        self.mode.compare_exchange(0, -1, SeqCst, SeqCst).is_ok()
    }

    #[inline]
    pub unsafe fn read_unlock(&self) {
        self.mode.fetch_sub(1, SeqCst);
    }

    #[inline]
    pub unsafe fn write_unlock(&self) {
        assert_eq!(self.mode.compare_exchange(-1, 0, SeqCst, SeqCst), Ok(-1));
    }
}
