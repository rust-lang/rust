#[cfg(test)]
mod tests;

use crate::num::NonZeroUsize;
use crate::sys_common::lazy_box::{LazyBox, LazyInit};

use super::waitqueue::{
    try_lock_or_false, NotifiedTcs, SpinMutex, SpinMutexGuard, WaitQueue, WaitVariable,
};
use crate::alloc::Layout;

struct AllocatedRwLock {
    readers: SpinMutex<WaitVariable<Option<NonZeroUsize>>>,
    writer: SpinMutex<WaitVariable<bool>>,
}

pub struct RwLock {
    inner: LazyBox<AllocatedRwLock>,
}

impl LazyInit for AllocatedRwLock {
    fn init() -> Box<Self> {
        Box::new(AllocatedRwLock {
            readers: SpinMutex::new(WaitVariable::new(None)),
            writer: SpinMutex::new(WaitVariable::new(false)),
        })
    }
}

// Check at compile time that RwLock's size and alignment matches the C definition
// in libunwind (see also `test_c_rwlock_initializer` in `tests`).
const _: () = {
    let rust = Layout::new::<RwLock>();
    let c = Layout::new::<*mut ()>();
    assert!(rust.size() == c.size());
    assert!(rust.align() == c.align());
};

impl RwLock {
    pub const fn new() -> RwLock {
        RwLock { inner: LazyBox::new() }
    }

    #[inline]
    pub fn read(&self) {
        let lock = &*self.inner;
        let mut rguard = lock.readers.lock();
        let wguard = lock.writer.lock();
        if *wguard.lock_var() || !wguard.queue_empty() {
            // Another thread has or is waiting for the write lock, wait
            drop(wguard);
            WaitQueue::wait(rguard, || {});
        // Another thread has passed the lock to us
        } else {
            // No waiting writers, acquire the read lock
            *rguard.lock_var_mut() =
                NonZeroUsize::new(rguard.lock_var().map_or(0, |n| n.get()) + 1);
        }
    }

    #[inline]
    pub unsafe fn try_read(&self) -> bool {
        let lock = &*self.inner;
        let mut rguard = try_lock_or_false!(lock.readers);
        let wguard = try_lock_or_false!(lock.writer);
        if *wguard.lock_var() || !wguard.queue_empty() {
            // Another thread has or is waiting for the write lock
            false
        } else {
            // No waiting writers, acquire the read lock
            *rguard.lock_var_mut() =
                NonZeroUsize::new(rguard.lock_var().map_or(0, |n| n.get()) + 1);
            true
        }
    }

    #[inline]
    pub fn write(&self) {
        let lock = &*self.inner;
        let rguard = lock.readers.lock();
        let mut wguard = lock.writer.lock();
        if *wguard.lock_var() || rguard.lock_var().is_some() {
            // Another thread has the lock, wait
            drop(rguard);
            WaitQueue::wait(wguard, || {});
        // Another thread has passed the lock to us
        } else {
            // We are just now obtaining the lock
            *wguard.lock_var_mut() = true;
        }
    }

    #[inline]
    pub fn try_write(&self) -> bool {
        let lock = &*self.inner;
        let rguard = try_lock_or_false!(lock.readers);
        let mut wguard = try_lock_or_false!(lock.writer);
        if *wguard.lock_var() || rguard.lock_var().is_some() {
            // Another thread has the lock
            false
        } else {
            // We are just now obtaining the lock
            *wguard.lock_var_mut() = true;
            true
        }
    }

    #[inline]
    unsafe fn __read_unlock(
        &self,
        mut rguard: SpinMutexGuard<'_, WaitVariable<Option<NonZeroUsize>>>,
        wguard: SpinMutexGuard<'_, WaitVariable<bool>>,
    ) {
        *rguard.lock_var_mut() = NonZeroUsize::new(rguard.lock_var().unwrap().get() - 1);
        if rguard.lock_var().is_some() {
            // There are other active readers
        } else {
            if let Ok(mut wguard) = WaitQueue::notify_one(wguard) {
                // A writer was waiting, pass the lock
                *wguard.lock_var_mut() = true;
                wguard.drop_after(rguard);
            } else {
                // No writers were waiting, the lock is released
                rtassert!(rguard.queue_empty());
            }
        }
    }

    #[inline]
    pub unsafe fn read_unlock(&self) {
        let lock = &*self.inner;
        let rguard = lock.readers.lock();
        let wguard = lock.writer.lock();
        unsafe { self.__read_unlock(rguard, wguard) };
    }

    #[inline]
    unsafe fn __write_unlock(
        &self,
        rguard: SpinMutexGuard<'_, WaitVariable<Option<NonZeroUsize>>>,
        wguard: SpinMutexGuard<'_, WaitVariable<bool>>,
    ) {
        match WaitQueue::notify_one(wguard) {
            Err(mut wguard) => {
                // No writers waiting, release the write lock
                *wguard.lock_var_mut() = false;
                if let Ok(mut rguard) = WaitQueue::notify_all(rguard) {
                    // One or more readers were waiting, pass the lock to them
                    if let NotifiedTcs::All { count } = rguard.notified_tcs() {
                        *rguard.lock_var_mut() = Some(count)
                    } else {
                        unreachable!() // called notify_all
                    }
                    rguard.drop_after(wguard);
                } else {
                    // No readers waiting, the lock is released
                }
            }
            Ok(wguard) => {
                // There was a thread waiting for write, just pass the lock
                wguard.drop_after(rguard);
            }
        }
    }

    #[inline]
    pub unsafe fn write_unlock(&self) {
        let lock = &*self.inner;
        let rguard = lock.readers.lock();
        let wguard = lock.writer.lock();
        unsafe { self.__write_unlock(rguard, wguard) };
    }

    // only used by __rust_rwlock_unlock below
    #[inline]
    #[cfg_attr(test, allow(dead_code))]
    unsafe fn unlock(&self) {
        let lock = &*self.inner;
        let rguard = lock.readers.lock();
        let wguard = lock.writer.lock();
        if *wguard.lock_var() == true {
            unsafe { self.__write_unlock(rguard, wguard) };
        } else {
            unsafe { self.__read_unlock(rguard, wguard) };
        }
    }
}

// The following functions are needed by libunwind. These symbols are named
// in pre-link args for the target specification, so keep that in sync.
#[cfg(not(test))]
const EINVAL: i32 = 22;

#[cfg(not(test))]
#[no_mangle]
pub unsafe extern "C" fn __rust_rwlock_rdlock(p: *mut RwLock) -> i32 {
    if p.is_null() {
        return EINVAL;
    }
    unsafe { (*p).read() };
    return 0;
}

#[cfg(not(test))]
#[no_mangle]
pub unsafe extern "C" fn __rust_rwlock_wrlock(p: *mut RwLock) -> i32 {
    if p.is_null() {
        return EINVAL;
    }
    unsafe { (*p).write() };
    return 0;
}

#[cfg(not(test))]
#[no_mangle]
pub unsafe extern "C" fn __rust_rwlock_unlock(p: *mut RwLock) -> i32 {
    if p.is_null() {
        return EINVAL;
    }
    unsafe { (*p).unlock() };
    return 0;
}
