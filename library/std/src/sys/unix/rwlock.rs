use crate::cell::UnsafeCell;
use crate::sync::atomic::{AtomicUsize, Ordering};

pub struct RWLock {
    inner: UnsafeCell<libc::pthread_rwlock_t>,
    write_locked: UnsafeCell<bool>, // guarded by the `inner` RwLock
    num_readers: AtomicUsize,
}

pub type MovableRWLock = Box<RWLock>;

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

        // According to POSIX, when a thread tries to acquire this read lock
        // while it already holds the write lock
        // (or vice versa, or tries to acquire the write lock twice),
        // "the call shall either deadlock or return [EDEADLK]"
        // (https://pubs.opengroup.org/onlinepubs/9699919799/functions/pthread_rwlock_wrlock.html,
        // https://pubs.opengroup.org/onlinepubs/9699919799/functions/pthread_rwlock_rdlock.html).
        // So, in principle, all we have to do here is check `r == 0` to be sure we properly
        // got the lock.
        //
        // However, (at least) glibc before version 2.25 does not conform to this spec,
        // and can return `r == 0` even when this thread already holds the write lock.
        // We thus check for this situation ourselves and panic when detecting that a thread
        // got the write lock more than once, or got a read and a write lock.
        if r == libc::EAGAIN {
            panic!("rwlock maximum reader count exceeded");
        } else if r == libc::EDEADLK || (r == 0 && *self.write_locked.get()) {
            // Above, we make sure to only access `write_locked` when `r == 0` to avoid
            // data races.
            if r == 0 {
                // `pthread_rwlock_rdlock` succeeded when it should not have.
                self.raw_unlock();
            }
            panic!("rwlock read lock would result in deadlock");
        } else {
            // According to POSIX, for a properly initialized rwlock this can only
            // return EAGAIN or EDEADLK or 0. We rely on that.
            debug_assert_eq!(r, 0);
            self.num_readers.fetch_add(1, Ordering::Relaxed);
        }
    }
    #[inline]
    pub unsafe fn try_read(&self) -> bool {
        let r = libc::pthread_rwlock_tryrdlock(self.inner.get());
        if r == 0 {
            if *self.write_locked.get() {
                // `pthread_rwlock_tryrdlock` succeeded when it should not have.
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
        // See comments above for why we check for EDEADLK and write_locked. For the same reason,
        // we also need to check that there are no readers (tracked in `num_readers`).
        if r == libc::EDEADLK
            || (r == 0 && *self.write_locked.get())
            || self.num_readers.load(Ordering::Relaxed) != 0
        {
            // Above, we make sure to only access `write_locked` when `r == 0` to avoid
            // data races.
            if r == 0 {
                // `pthread_rwlock_wrlock` succeeded when it should not have.
                self.raw_unlock();
            }
            panic!("rwlock write lock would result in deadlock");
        } else {
            // According to POSIX, for a properly initialized rwlock this can only
            // return EDEADLK or 0. We rely on that.
            debug_assert_eq!(r, 0);
        }
        *self.write_locked.get() = true;
    }
    #[inline]
    pub unsafe fn try_write(&self) -> bool {
        let r = libc::pthread_rwlock_trywrlock(self.inner.get());
        if r == 0 {
            if *self.write_locked.get() || self.num_readers.load(Ordering::Relaxed) != 0 {
                // `pthread_rwlock_trywrlock` succeeded when it should not have.
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
        // On DragonFly pthread_rwlock_destroy() returns EINVAL if called on a
        // rwlock that was just initialized with
        // libc::PTHREAD_RWLOCK_INITIALIZER. Once it is used (locked/unlocked)
        // or pthread_rwlock_init() is called, this behaviour no longer occurs.
        if cfg!(target_os = "dragonfly") {
            debug_assert!(r == 0 || r == libc::EINVAL);
        } else {
            debug_assert_eq!(r, 0);
        }
    }
}
