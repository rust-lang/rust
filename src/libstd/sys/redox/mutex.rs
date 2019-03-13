use crate::cell::UnsafeCell;
use crate::intrinsics::{atomic_cxchg, atomic_xchg};
use crate::ptr;

use crate::sys::syscall::{futex, getpid, FUTEX_WAIT, FUTEX_WAKE};

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
    /// Creates a new mutex.
    pub const fn new() -> Self {
        Mutex {
            lock: UnsafeCell::new(0),
        }
    }

    #[inline]
    pub unsafe fn init(&self) {
        *self.lock.get() = 0;
    }

    /// Try to lock the mutex
    #[inline]
    pub unsafe fn try_lock(&self) -> bool {
        mutex_try_lock(self.lock.get())
    }

    /// Lock the mutex
    #[inline]
    pub unsafe fn lock(&self) {
        mutex_lock(self.lock.get());
    }

    /// Unlock the mutex
    #[inline]
    pub unsafe fn unlock(&self) {
        mutex_unlock(self.lock.get());
    }

    #[inline]
    pub unsafe fn destroy(&self) {
        *self.lock.get() = 0;
    }
}

unsafe impl Send for Mutex {}

unsafe impl Sync for Mutex {}

pub struct ReentrantMutex {
    pub lock: UnsafeCell<i32>,
    pub owner: UnsafeCell<usize>,
    pub own_count: UnsafeCell<usize>,
}

impl ReentrantMutex {
    pub const fn uninitialized() -> Self {
        ReentrantMutex {
            lock: UnsafeCell::new(0),
            owner: UnsafeCell::new(0),
            own_count: UnsafeCell::new(0),
        }
    }

    #[inline]
    pub unsafe fn init(&mut self) {
        *self.lock.get() = 0;
        *self.owner.get() = 0;
        *self.own_count.get() = 0;
    }

    /// Try to lock the mutex
    #[inline]
    pub unsafe fn try_lock(&self) -> bool {
        let pid = getpid().unwrap();
        if *self.own_count.get() > 0 && *self.owner.get() == pid {
            *self.own_count.get() += 1;
            true
        } else {
            if mutex_try_lock(self.lock.get()) {
                *self.owner.get() = pid;
                *self.own_count.get() = 1;
                true
            } else {
                false
            }
        }
    }

    /// Lock the mutex
    #[inline]
    pub unsafe fn lock(&self) {
        let pid = getpid().unwrap();
        if *self.own_count.get() > 0 && *self.owner.get() == pid {
            *self.own_count.get() += 1;
        } else {
            mutex_lock(self.lock.get());
            *self.owner.get() = pid;
            *self.own_count.get() = 1;
        }
    }

    /// Unlock the mutex
    #[inline]
    pub unsafe fn unlock(&self) {
        let pid = getpid().unwrap();
        if *self.own_count.get() > 0 && *self.owner.get() == pid {
            *self.own_count.get() -= 1;
            if *self.own_count.get() == 0 {
                *self.owner.get() = 0;
                mutex_unlock(self.lock.get());
            }
        }
    }

    #[inline]
    pub unsafe fn destroy(&self) {
        *self.lock.get() = 0;
        *self.owner.get() = 0;
        *self.own_count.get() = 0;
    }
}

unsafe impl Send for ReentrantMutex {}

unsafe impl Sync for ReentrantMutex {}
