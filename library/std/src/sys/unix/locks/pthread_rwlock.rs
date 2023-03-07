use crate::cell::UnsafeCell;
use crate::mem::forget;
use crate::sync::atomic::{AtomicUsize, Ordering};
use crate::sys_common::lazy_box::{LazyBox, LazyInit};

struct AllocatedRwLock {
    inner: UnsafeCell<libc::pthread_rwlock_t>,
    write_locked: UnsafeCell<bool>, // guarded by the `inner` RwLock
    num_readers: AtomicUsize,
}

unsafe impl Send for AllocatedRwLock {}
unsafe impl Sync for AllocatedRwLock {}

pub struct RwLock {
    inner: LazyBox<AllocatedRwLock>,
}

impl LazyInit for AllocatedRwLock {
    fn init() -> Box<Self> {
        Box::new(AllocatedRwLock {
            inner: UnsafeCell::new(libc::PTHREAD_RWLOCK_INITIALIZER),
            write_locked: UnsafeCell::new(false),
            num_readers: AtomicUsize::new(0),
        })
    }

    fn destroy(mut rwlock: Box<Self>) {
        // We're not allowed to pthread_rwlock_destroy a locked rwlock,
        // so check first if it's unlocked.
        if *rwlock.write_locked.get_mut() || *rwlock.num_readers.get_mut() != 0 {
            // The rwlock is locked. This happens if a RwLock{Read,Write}Guard is leaked.
            // In this case, we just leak the RwLock too.
            forget(rwlock);
        }
    }

    fn cancel_init(_: Box<Self>) {
        // In this case, we can just drop it without any checks,
        // since it cannot have been locked yet.
    }
}

impl AllocatedRwLock {
    #[inline]
    unsafe fn raw_unlock(&self) {
        let r = libc::pthread_rwlock_unlock(self.inner.get());
        debug_assert_eq!(r, 0);
    }
}

impl Drop for AllocatedRwLock {
    fn drop(&mut self) {
        let r = unsafe { libc::pthread_rwlock_destroy(self.inner.get()) };
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

impl RwLock {
    #[inline]
    pub const fn new() -> RwLock {
        RwLock { inner: LazyBox::new() }
    }

    #[inline]
    pub fn read(&self) {
        let lock = &*self.inner;
        let r = unsafe { libc::pthread_rwlock_rdlock(lock.inner.get()) };

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
        } else if r == libc::EDEADLK || (r == 0 && unsafe { *lock.write_locked.get() }) {
            // Above, we make sure to only access `write_locked` when `r == 0` to avoid
            // data races.
            if r == 0 {
                // `pthread_rwlock_rdlock` succeeded when it should not have.
                unsafe {
                    lock.raw_unlock();
                }
            }
            panic!("rwlock read lock would result in deadlock");
        } else {
            // POSIX does not make guarantees about all the errors that may be returned.
            // See issue #94705 for more details.
            assert_eq!(r, 0, "unexpected error during rwlock read lock: {:?}", r);
            lock.num_readers.fetch_add(1, Ordering::Relaxed);
        }
    }

    #[inline]
    pub fn try_read(&self) -> bool {
        let lock = &*self.inner;
        let r = unsafe { libc::pthread_rwlock_tryrdlock(lock.inner.get()) };
        if r == 0 {
            if unsafe { *lock.write_locked.get() } {
                // `pthread_rwlock_tryrdlock` succeeded when it should not have.
                unsafe {
                    lock.raw_unlock();
                }
                false
            } else {
                lock.num_readers.fetch_add(1, Ordering::Relaxed);
                true
            }
        } else {
            false
        }
    }

    #[inline]
    pub fn write(&self) {
        let lock = &*self.inner;
        let r = unsafe { libc::pthread_rwlock_wrlock(lock.inner.get()) };
        // See comments above for why we check for EDEADLK and write_locked. For the same reason,
        // we also need to check that there are no readers (tracked in `num_readers`).
        if r == libc::EDEADLK
            || (r == 0 && unsafe { *lock.write_locked.get() })
            || lock.num_readers.load(Ordering::Relaxed) != 0
        {
            // Above, we make sure to only access `write_locked` when `r == 0` to avoid
            // data races.
            if r == 0 {
                // `pthread_rwlock_wrlock` succeeded when it should not have.
                unsafe {
                    lock.raw_unlock();
                }
            }
            panic!("rwlock write lock would result in deadlock");
        } else {
            // According to POSIX, for a properly initialized rwlock this can only
            // return EDEADLK or 0. We rely on that.
            debug_assert_eq!(r, 0);
        }

        unsafe {
            *lock.write_locked.get() = true;
        }
    }

    #[inline]
    pub unsafe fn try_write(&self) -> bool {
        let lock = &*self.inner;
        let r = libc::pthread_rwlock_trywrlock(lock.inner.get());
        if r == 0 {
            if *lock.write_locked.get() || lock.num_readers.load(Ordering::Relaxed) != 0 {
                // `pthread_rwlock_trywrlock` succeeded when it should not have.
                lock.raw_unlock();
                false
            } else {
                *lock.write_locked.get() = true;
                true
            }
        } else {
            false
        }
    }

    #[inline]
    pub unsafe fn read_unlock(&self) {
        let lock = &*self.inner;
        debug_assert!(!*lock.write_locked.get());
        lock.num_readers.fetch_sub(1, Ordering::Relaxed);
        lock.raw_unlock();
    }

    #[inline]
    pub unsafe fn write_unlock(&self) {
        let lock = &*self.inner;
        debug_assert_eq!(lock.num_readers.load(Ordering::Relaxed), 0);
        debug_assert!(*lock.write_locked.get());
        *lock.write_locked.get() = false;
        lock.raw_unlock();
    }
}
