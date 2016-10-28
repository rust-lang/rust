use cell::UnsafeCell;
use intrinsics::{atomic_cxchg, atomic_xchg};
use ptr;

use libc::{futex, FUTEX_WAIT, FUTEX_WAKE};

pub unsafe fn mutex_try_lock(m: *mut i32) -> bool {
    atomic_cxchg(m, 0, 1).0 == 0
}

pub unsafe fn mutex_lock(m: *mut i32) {
    let mut c = 0;
    //Set to larger value for longer spin test
    for _i in 0..100 {
        c = atomic_cxchg(m, 0, 1).0;
        if c == 0 {
            break;
        }
        //cpu_relax()
    }
    if c == 1 {
        c = atomic_xchg(m, 2);
    }
    while c != 0 {
        let _ = futex(m, FUTEX_WAIT, 2, 0, ptr::null_mut());
        c = atomic_xchg(m, 2);
    }
}

pub unsafe fn mutex_unlock(m: *mut i32) {
    if *m == 2 {
        *m = 0;
    } else if atomic_xchg(m, 0) == 1 {
        return;
    }
    //Set to larger value for longer spin test
    for _i in 0..100 {
        if *m != 0 {
            if atomic_cxchg(m, 1, 2).0 != 0 {
                return;
            }
        }
        //cpu_relax()
    }
    let _ = futex(m, FUTEX_WAKE, 1, 0, ptr::null_mut());
}

pub struct Mutex {
    pub lock: UnsafeCell<i32>,
}

impl Mutex {
    /// Create a new mutex.
    pub const fn new() -> Self {
        Mutex {
            lock: UnsafeCell::new(0),
        }
    }

    pub unsafe fn init(&self) {

    }

    /// Try to lock the mutex
    pub unsafe fn try_lock(&self) -> bool {
        mutex_try_lock(self.lock.get())
    }

    /// Lock the mutex
    pub unsafe fn lock(&self) {
        mutex_lock(self.lock.get());
    }

    /// Unlock the mutex
    pub unsafe fn unlock(&self) {
        mutex_unlock(self.lock.get());
    }

    pub unsafe fn destroy(&self) {

    }
}

unsafe impl Send for Mutex {}

unsafe impl Sync for Mutex {}

pub struct ReentrantMutex {
    pub lock: UnsafeCell<i32>,
}

impl ReentrantMutex {
    pub const fn uninitialized() -> Self {
        ReentrantMutex {
            lock: UnsafeCell::new(0),
        }
    }

    pub unsafe fn init(&mut self) {

    }

    /// Try to lock the mutex
    pub unsafe fn try_lock(&self) -> bool {
        mutex_try_lock(self.lock.get())
    }

    /// Lock the mutex
    pub unsafe fn lock(&self) {
        mutex_lock(self.lock.get());
    }

    /// Unlock the mutex
    pub unsafe fn unlock(&self) {
        mutex_unlock(self.lock.get());
    }

    pub unsafe fn destroy(&self) {

    }
}

unsafe impl Send for ReentrantMutex {}

unsafe impl Sync for ReentrantMutex {}
