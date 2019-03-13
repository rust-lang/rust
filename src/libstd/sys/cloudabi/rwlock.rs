use crate::cell::UnsafeCell;
use crate::mem;
use crate::sync::atomic::{AtomicU32, Ordering};
use crate::sys::cloudabi::abi;

extern "C" {
    #[thread_local]
    static __pthread_thread_id: abi::tid;
}

#[thread_local]
static mut RDLOCKS_ACQUIRED: u32 = 0;

pub struct RWLock {
    lock: UnsafeCell<AtomicU32>,
}

pub unsafe fn raw(r: &RWLock) -> *mut AtomicU32 {
    r.lock.get()
}

unsafe impl Send for RWLock {}
unsafe impl Sync for RWLock {}

const NEW: RWLock = RWLock {
    lock: UnsafeCell::new(AtomicU32::new(abi::LOCK_UNLOCKED.0)),
};

impl RWLock {
    pub const fn new() -> RWLock {
        NEW
    }

    pub unsafe fn try_read(&self) -> bool {
        let lock = self.lock.get();
        let mut old = abi::LOCK_UNLOCKED.0;
        while let Err(cur) =
            (*lock).compare_exchange_weak(old, old + 1, Ordering::Acquire, Ordering::Relaxed)
        {
            if (cur & abi::LOCK_WRLOCKED.0) != 0 {
                // Another thread already has a write lock.
                assert_ne!(
                    old & !abi::LOCK_KERNEL_MANAGED.0,
                    __pthread_thread_id.0 | abi::LOCK_WRLOCKED.0,
                    "Attempted to acquire a read lock while holding a write lock"
                );
                return false;
            } else if (old & abi::LOCK_KERNEL_MANAGED.0) != 0 && RDLOCKS_ACQUIRED == 0 {
                // Lock has threads waiting for the lock. Only acquire
                // the lock if we have already acquired read locks. In
                // that case, it is justified to acquire this lock to
                // prevent a deadlock.
                return false;
            }
            old = cur;
        }

        RDLOCKS_ACQUIRED += 1;
        true
    }

    pub unsafe fn read(&self) {
        if !self.try_read() {
            // Call into the kernel to acquire a read lock.
            let lock = self.lock.get();
            let subscription = abi::subscription {
                type_: abi::eventtype::LOCK_RDLOCK,
                union: abi::subscription_union {
                    lock: abi::subscription_lock {
                        lock: lock as *mut abi::lock,
                        lock_scope: abi::scope::PRIVATE,
                    },
                },
                ..mem::zeroed()
            };
            let mut event: abi::event = mem::uninitialized();
            let mut nevents: usize = mem::uninitialized();
            let ret = abi::poll(&subscription, &mut event, 1, &mut nevents);
            assert_eq!(ret, abi::errno::SUCCESS, "Failed to acquire read lock");
            assert_eq!(
                event.error,
                abi::errno::SUCCESS,
                "Failed to acquire read lock"
            );

            RDLOCKS_ACQUIRED += 1;
        }
    }

    pub unsafe fn read_unlock(&self) {
        // Perform a read unlock. We can do this in userspace, except when
        // other threads are blocked and we are performing the last unlock.
        // In that case, call into the kernel.
        //
        // Other threads may attempt to increment the read lock count,
        // meaning that the call into the kernel could be spurious. To
        // prevent this from happening, upgrade to a write lock first. This
        // allows us to call into the kernel, having the guarantee that the
        // lock value will not change in the meantime.
        assert!(RDLOCKS_ACQUIRED > 0, "Bad lock count");
        let mut old = 1;
        loop {
            let lock = self.lock.get();
            if old == 1 | abi::LOCK_KERNEL_MANAGED.0 {
                // Last read lock while threads are waiting. Attempt to upgrade
                // to a write lock before calling into the kernel to unlock.
                if let Err(cur) = (*lock).compare_exchange_weak(
                    old,
                    __pthread_thread_id.0 | abi::LOCK_WRLOCKED.0 | abi::LOCK_KERNEL_MANAGED.0,
                    Ordering::Acquire,
                    Ordering::Relaxed,
                ) {
                    old = cur;
                } else {
                    // Call into the kernel to unlock.
                    let ret = abi::lock_unlock(lock as *mut abi::lock, abi::scope::PRIVATE);
                    assert_eq!(ret, abi::errno::SUCCESS, "Failed to write unlock a rwlock");
                    break;
                }
            } else {
                // No threads waiting or not the last read lock. Just decrement
                // the read lock count.
                assert_ne!(
                    old & !abi::LOCK_KERNEL_MANAGED.0,
                    0,
                    "This rwlock is not locked"
                );
                assert_eq!(
                    old & abi::LOCK_WRLOCKED.0,
                    0,
                    "Attempted to read-unlock a write-locked rwlock"
                );
                if let Err(cur) = (*lock).compare_exchange_weak(
                    old,
                    old - 1,
                    Ordering::Acquire,
                    Ordering::Relaxed,
                ) {
                    old = cur;
                } else {
                    break;
                }
            }
        }

        RDLOCKS_ACQUIRED -= 1;
    }

    pub unsafe fn try_write(&self) -> bool {
        // Attempt to acquire the lock.
        let lock = self.lock.get();
        if let Err(old) = (*lock).compare_exchange(
            abi::LOCK_UNLOCKED.0,
            __pthread_thread_id.0 | abi::LOCK_WRLOCKED.0,
            Ordering::Acquire,
            Ordering::Relaxed,
        ) {
            // Failure. Crash upon recursive acquisition.
            assert_ne!(
                old & !abi::LOCK_KERNEL_MANAGED.0,
                __pthread_thread_id.0 | abi::LOCK_WRLOCKED.0,
                "Attempted to recursive write-lock a rwlock",
            );
            false
        } else {
            // Success.
            true
        }
    }

    pub unsafe fn write(&self) {
        if !self.try_write() {
            // Call into the kernel to acquire a write lock.
            let lock = self.lock.get();
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
            let mut event: abi::event = mem::uninitialized();
            let mut nevents: usize = mem::uninitialized();
            let ret = abi::poll(&subscription, &mut event, 1, &mut nevents);
            assert_eq!(ret, abi::errno::SUCCESS, "Failed to acquire write lock");
            assert_eq!(
                event.error,
                abi::errno::SUCCESS,
                "Failed to acquire write lock"
            );
        }
    }

    pub unsafe fn write_unlock(&self) {
        let lock = self.lock.get();
        assert_eq!(
            (*lock).load(Ordering::Relaxed) & !abi::LOCK_KERNEL_MANAGED.0,
            __pthread_thread_id.0 | abi::LOCK_WRLOCKED.0,
            "This rwlock is not write-locked by this thread"
        );

        if !(*lock)
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
            let ret = abi::lock_unlock(lock as *mut abi::lock, abi::scope::PRIVATE);
            assert_eq!(ret, abi::errno::SUCCESS, "Failed to write unlock a rwlock");
        }
    }

    pub unsafe fn destroy(&self) {
        let lock = self.lock.get();
        assert_eq!(
            (*lock).load(Ordering::Relaxed),
            abi::LOCK_UNLOCKED.0,
            "Attempted to destroy locked rwlock"
        );
    }
}
