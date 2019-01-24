use alloc::{self, Layout};
use num::NonZeroUsize;
use slice;
use str;

use super::waitqueue::{
    try_lock_or_false, NotifiedTcs, SpinMutex, SpinMutexGuard, WaitQueue, WaitVariable,
};
use mem;

pub struct RWLock {
    readers: SpinMutex<WaitVariable<Option<NonZeroUsize>>>,
    writer: SpinMutex<WaitVariable<bool>>,
}

// Below is to check at compile time, that RWLock has size of 128 bytes.
#[allow(dead_code)]
unsafe fn rw_lock_size_assert(r: RWLock) {
    mem::transmute::<RWLock, [u8; 128]>(r);
}

//unsafe impl Send for RWLock {}
//unsafe impl Sync for RWLock {} // FIXME

impl RWLock {
    pub const fn new() -> RWLock {
        RWLock {
            readers: SpinMutex::new(WaitVariable::new(None)),
            writer: SpinMutex::new(WaitVariable::new(false)),
        }
    }

    #[inline]
    pub unsafe fn read(&self) {
        let mut rguard = self.readers.lock();
        let wguard = self.writer.lock();
        if *wguard.lock_var() || !wguard.queue_empty() {
            // Another thread has or is waiting for the write lock, wait
            drop(wguard);
            WaitQueue::wait(rguard);
            // Another thread has passed the lock to us
        } else {
            // No waiting writers, acquire the read lock
            *rguard.lock_var_mut() =
                NonZeroUsize::new(rguard.lock_var().map_or(0, |n| n.get()) + 1);
        }
    }

    #[inline]
    pub unsafe fn try_read(&self) -> bool {
        let mut rguard = try_lock_or_false!(self.readers);
        let wguard = try_lock_or_false!(self.writer);
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
    pub unsafe fn write(&self) {
        let rguard = self.readers.lock();
        let mut wguard = self.writer.lock();
        if *wguard.lock_var() || rguard.lock_var().is_some() {
            // Another thread has the lock, wait
            drop(rguard);
            WaitQueue::wait(wguard);
            // Another thread has passed the lock to us
        } else {
            // We are just now obtaining the lock
            *wguard.lock_var_mut() = true;
        }
    }

    #[inline]
    pub unsafe fn try_write(&self) -> bool {
        let rguard = try_lock_or_false!(self.readers);
        let mut wguard = try_lock_or_false!(self.writer);
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
        mut rguard: SpinMutexGuard<WaitVariable<Option<NonZeroUsize>>>,
        wguard: SpinMutexGuard<WaitVariable<bool>>,
    ) {
        *rguard.lock_var_mut() = NonZeroUsize::new(rguard.lock_var().unwrap().get() - 1);
        if rguard.lock_var().is_some() {
            // There are other active readers
        } else {
            if let Ok(mut wguard) = WaitQueue::notify_one(wguard) {
                // A writer was waiting, pass the lock
                *wguard.lock_var_mut() = true;
            } else {
                // No writers were waiting, the lock is released
                assert!(rguard.queue_empty());
            }
        }
    }

    #[inline]
    pub unsafe fn read_unlock(&self) {
        let rguard = self.readers.lock();
        let wguard = self.writer.lock();
        self.__read_unlock(rguard, wguard);
    }

    #[inline]
    unsafe fn __write_unlock(
        &self,
        rguard: SpinMutexGuard<WaitVariable<Option<NonZeroUsize>>>,
        wguard: SpinMutexGuard<WaitVariable<bool>>,
    ) {
        if let Err(mut wguard) = WaitQueue::notify_one(wguard) {
            // No writers waiting, release the write lock
            *wguard.lock_var_mut() = false;
            if let Ok(mut rguard) = WaitQueue::notify_all(rguard) {
                // One or more readers were waiting, pass the lock to them
                if let NotifiedTcs::All { count } = rguard.notified_tcs() {
                    *rguard.lock_var_mut() = Some(count)
                } else {
                    unreachable!() // called notify_all
                }
            } else {
                // No readers waiting, the lock is released
            }
        } else {
            // There was a thread waiting for write, just pass the lock
        }
    }

    #[inline]
    pub unsafe fn write_unlock(&self) {
        let rguard = self.readers.lock();
        let wguard = self.writer.lock();
        self.__write_unlock(rguard, wguard);
    }

    // only used by __rust_rwlock_unlock below
    #[inline]
    unsafe fn unlock(&self) {
        let rguard = self.readers.lock();
        let wguard = self.writer.lock();
        if *wguard.lock_var() == true {
            self.__write_unlock(rguard, wguard);
        } else {
            self.__read_unlock(rguard, wguard);
        }
    }

    #[inline]
    pub unsafe fn destroy(&self) {}
}

const EINVAL: i32 = 22;

// used by libunwind port
#[no_mangle]
pub unsafe extern "C" fn __rust_rwlock_rdlock(p: *mut RWLock) -> i32 {
    if p.is_null() {
        return EINVAL;
    }
    (*p).read();
    return 0;
}

#[no_mangle]
pub unsafe extern "C" fn __rust_rwlock_wrlock(p: *mut RWLock) -> i32 {
    if p.is_null() {
        return EINVAL;
    }
    (*p).write();
    return 0;
}
#[no_mangle]
pub unsafe extern "C" fn __rust_rwlock_unlock(p: *mut RWLock) -> i32 {
    if p.is_null() {
        return EINVAL;
    }
    (*p).unlock();
    return 0;
}

// the following functions are also used by the libunwind port. They're
// included here to make sure parallel codegen and LTO don't mess things up.
#[no_mangle]
pub unsafe extern "C" fn __rust_print_err(m: *mut u8, s: i32) {
    if s < 0 {
        return;
    }
    let buf = slice::from_raw_parts(m as *const u8, s as _);
    if let Ok(s) = str::from_utf8(&buf[..buf.iter().position(|&b| b == 0).unwrap_or(buf.len())]) {
        eprint!("{}", s);
    }
}

#[no_mangle]
pub unsafe extern "C" fn __rust_abort() {
    ::sys::abort_internal();
}

#[no_mangle]
pub unsafe extern "C" fn __rust_c_alloc(size: usize, align: usize) -> *mut u8 {
    alloc::alloc(Layout::from_size_align_unchecked(size, align))
}

#[no_mangle]
pub unsafe extern "C" fn __rust_c_dealloc(ptr: *mut u8, size: usize, align: usize) {
    alloc::dealloc(ptr, Layout::from_size_align_unchecked(size, align))
}

#[cfg(test)]
mod tests {

    use super::*;
    use core::array::FixedSizeArray;
    use mem::MaybeUninit;
    use {mem, ptr};

    // The below test verifies that the bytes of initialized RWLock are the ones
    // we use in libunwind.
    // If they change we need to update src/UnwindRustSgx.h in libunwind.
    #[test]
    fn test_c_rwlock_initializer() {
        const RWLOCK_INIT: &[u8] = &[
            0x1, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0,
            0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0,
            0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0,
            0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0,
            0x3, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0,
            0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0,
            0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0,
            0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0,
            0x1, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0,
            0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0,
            0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0,
            0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0,
            0x3, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0,
            0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0,
            0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0,
            0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0,
        ];

        let mut init = MaybeUninit::<RWLock>::zeroed();
        init.set(RWLock::new());
        assert_eq!(
            mem::transmute::<_, [u8; 128]>(init.into_inner()).as_slice(),
            RWLOCK_INIT
        );
    }
}
