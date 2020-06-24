use crate::cell::UnsafeCell;
use crate::sync::atomic::{AtomicUsize, Ordering};

pub struct RWLock {
    inner: UnsafeCell<libc::pthread_rwlock_t>,
    write_locked: UnsafeCell<bool>,
    num_readers: AtomicUsize,
}

unsafe impl Send for RWLock {}
unsafe impl Sync for RWLock {}

impl RWLock {
    pub const fn new() -> RWLock {
        RWLock {
            inner: UnsafeCell::new(libc::PTHREAD_RWLOCK_INITIALIZER),
            write_locked: UnsafeCell::new(false),
            num_readers: AtomicUsize::new(0),
        }
    }

    #[inline]
    pub unsafe fn read(&self) {
        let r = libc::pthread_rwlock_rdlock(self.inner.get());
        if r == libc::EAGAIN {
            panic!("rwlock maximum reader count exceeded");
        } else if r == libc::EDEADLK || (r == 0 && *self.write_locked.get()) {
            if r == 0 {
                self.raw_unlock();
            }
            panic!("rwlock read lock would result in deadlock");
        } else {
            debug_assert_eq!(r, 0);
            self.num_readers.fetch_add(1, Ordering::Relaxed);
        }
    }

    #[inline]
    pub unsafe fn try_read(&self) -> bool {
        let r = libc::pthread_rwlock_tryrdlock(self.inner.get());
        if r == 0 {
            if *self.write_locked.get() {
                self.raw_unlock();
                false
            } else {
                self.num_readers.fetch_add(1, Ordering::Relaxed);
                true
            }
        } else {
            false
        }
    }

    #[inline]
    pub unsafe fn write(&self) {
        let r = libc::pthread_rwlock_wrlock(self.inner.get());
        // See comments above for why we check for EDEADLK and write_locked. We
        // also need to check that num_readers is 0.
        if r == libc::EDEADLK
            || *self.write_locked.get()
            || self.num_readers.load(Ordering::Relaxed) != 0
        {
            if r == 0 {
                self.raw_unlock();
            }
            panic!("rwlock write lock would result in deadlock");
        } else {
            debug_assert_eq!(r, 0);
        }
        *self.write_locked.get() = true;
    }

    #[inline]
    pub unsafe fn try_write(&self) -> bool {
        let r = libc::pthread_rwlock_trywrlock(self.inner.get());
        if r == 0 {
            if *self.write_locked.get() || self.num_readers.load(Ordering::Relaxed) != 0 {
                self.raw_unlock();
                false
            } else {
                *self.write_locked.get() = true;
                true
            }
        } else {
            false
        }
    }

    #[inline]
    unsafe fn raw_unlock(&self) {
        let r = libc::pthread_rwlock_unlock(self.inner.get());
        debug_assert_eq!(r, 0);
    }

    #[inline]
    pub unsafe fn read_unlock(&self) {
        debug_assert!(!*self.write_locked.get());
        self.num_readers.fetch_sub(1, Ordering::Relaxed);
        self.raw_unlock();
    }

    #[inline]
    pub unsafe fn write_unlock(&self) {
        debug_assert_eq!(self.num_readers.load(Ordering::Relaxed), 0);
        debug_assert!(*self.write_locked.get());
        *self.write_locked.get() = false;
        self.raw_unlock();
    }
    #[inline]
    pub unsafe fn destroy(&self) {
        let r = libc::pthread_rwlock_destroy(self.inner.get());
        debug_assert_eq!(r, 0);
    }
}
