use crate::cell::Cell;
use crate::mem;
use crate::mem::MaybeUninit;
use crate::sync::atomic::{AtomicU32, Ordering};
use crate::sys::cloudabi::abi;
use crate::sys::rwlock::{self, RWLock};

extern "C" {
    #[thread_local]
    static __pthread_thread_id: abi::tid;
}

// Implement Mutex using an RWLock. This doesn't introduce any
// performance overhead in this environment, as the operations would be
// implemented identically.
pub struct Mutex(RWLock);

pub type MovableMutex = Mutex;

pub unsafe fn raw(m: &Mutex) -> &AtomicU32 {
    rwlock::raw(&m.0)
}

impl Mutex {
    pub const fn new() -> Mutex {
        Mutex(RWLock::new())
    }

    pub unsafe fn init(&mut self) {
        // This function should normally reinitialize the mutex after
        // moving it to a different memory address. This implementation
        // does not require adjustments after moving.
    }

    pub unsafe fn try_lock(&self) -> bool {
        self.0.try_write()
    }

    pub unsafe fn lock(&self) {
        self.0.write()
    }

    pub unsafe fn unlock(&self) {
        self.0.write_unlock()
    }

    pub unsafe fn destroy(&self) {
        self.0.destroy()
    }
}

pub struct ReentrantMutex {
    lock: AtomicU32,
    recursion: Cell<u32>,
}

unsafe impl Send for ReentrantMutex {}
unsafe impl Sync for ReentrantMutex {}

impl ReentrantMutex {
    pub const unsafe fn uninitialized() -> ReentrantMutex {
        ReentrantMutex { lock: AtomicU32::new(abi::LOCK_UNLOCKED.0), recursion: Cell::new(0) }
    }

    pub unsafe fn init(&self) {}

    pub unsafe fn try_lock(&self) -> bool {
        // Attempt to acquire the lock.
        if let Err(old) = self.lock.compare_exchange(
            abi::LOCK_UNLOCKED.0,
            __pthread_thread_id.0 | abi::LOCK_WRLOCKED.0,
            Ordering::Acquire,
            Ordering::Relaxed,
        ) {
            // If we fail to acquire the lock, it may be the case
            // that we've already acquired it and may need to recurse.
            if old & !abi::LOCK_KERNEL_MANAGED.0 == __pthread_thread_id.0 | abi::LOCK_WRLOCKED.0 {
                self.recursion.set(self.recursion.get() + 1);
                true
            } else {
                false
            }
        } else {
            // Success.
            assert_eq!(self.recursion.get(), 0, "Mutex has invalid recursion count");
            true
        }
    }

    pub unsafe fn lock(&self) {
        if !self.try_lock() {
            // Call into the kernel to acquire a write lock.
            let lock = &self.lock as *const AtomicU32;
            let subscription = abi::subscription {
                type_: abi::eventtype::LOCK_WRLOCK,
                union: abi::subscription_union {
                    lock: abi::subscription_lock {
                        lock: lock as *mut abi::lock,
                        lock_scope: abi::scope::PRIVATE,
                    },
                },
                ..mem::zeroed()
            };
            let mut event = MaybeUninit::<abi::event>::uninit();
            let mut nevents = MaybeUninit::<usize>::uninit();
            // SAFE: The caller must to ensure that `event` and `nevents` are initialized.
            let ret =
                unsafe { abi::poll(&subscription, event.as_mut_ptr(), 1, nevents.as_mut_ptr()) };
            assert_eq!(ret, abi::errno::SUCCESS, "Failed to acquire mutex");
            let event = event.assume_init();
            assert_eq!(event.error, abi::errno::SUCCESS, "Failed to acquire mutex");
        }
    }

    pub unsafe fn unlock(&self) {
        assert_eq!(
            self.lock.load(Ordering::Relaxed) & !abi::LOCK_KERNEL_MANAGED.0,
            __pthread_thread_id.0 | abi::LOCK_WRLOCKED.0,
            "This mutex is locked by a different thread"
        );

        let r = self.recursion.get();
        if r > 0 {
            self.recursion.set(r - 1);
        } else if !self
            .lock
            .compare_exchange(
                __pthread_thread_id.0 | abi::LOCK_WRLOCKED.0,
                abi::LOCK_UNLOCKED.0,
                Ordering::Release,
                Ordering::Relaxed,
            )
            .is_ok()
        {
            // Lock is managed by kernelspace. Call into the kernel
            // to unblock waiting threads.
            let ret = abi::lock_unlock(
                &self.lock as *const AtomicU32 as *mut abi::lock,
                abi::scope::PRIVATE,
            );
            assert_eq!(ret, abi::errno::SUCCESS, "Failed to unlock a mutex");
        }
    }

    pub unsafe fn destroy(&self) {
        assert_eq!(
            self.lock.load(Ordering::Relaxed),
            abi::LOCK_UNLOCKED.0,
            "Attempted to destroy locked mutex"
        );
        assert_eq!(self.recursion.get(), 0, "Recursion counter invalid");
    }
}
