use crate::cell::UnsafeCell;
use crate::io::Error;
use crate::mem::{MaybeUninit, forget};
use crate::sys::cvt_nz;
use crate::sys::sync::OnceBox;

struct AllocatedMutex(UnsafeCell<libc::pthread_mutex_t>);

pub struct Mutex {
    inner: OnceBox<AllocatedMutex>,
}

unsafe impl Send for AllocatedMutex {}
unsafe impl Sync for AllocatedMutex {}

impl AllocatedMutex {
    fn new() -> Box<Self> {
        let mutex = Box::new(AllocatedMutex(UnsafeCell::new(libc::PTHREAD_MUTEX_INITIALIZER)));

        // Issue #33770
        //
        // A pthread mutex initialized with PTHREAD_MUTEX_INITIALIZER will have
        // a type of PTHREAD_MUTEX_DEFAULT, which has undefined behavior if you
        // try to re-lock it from the same thread when you already hold a lock
        // (https://pubs.opengroup.org/onlinepubs/9699919799/functions/pthread_mutex_init.html).
        // This is the case even if PTHREAD_MUTEX_DEFAULT == PTHREAD_MUTEX_NORMAL
        // (https://github.com/rust-lang/rust/issues/33770#issuecomment-220847521) -- in that
        // case, `pthread_mutexattr_settype(PTHREAD_MUTEX_DEFAULT)` will of course be the same
        // as setting it to `PTHREAD_MUTEX_NORMAL`, but not setting any mode will result in
        // a Mutex where re-locking is UB.
        //
        // In practice, glibc takes advantage of this undefined behavior to
        // implement hardware lock elision, which uses hardware transactional
        // memory to avoid acquiring the lock. While a transaction is in
        // progress, the lock appears to be unlocked. This isn't a problem for
        // other threads since the transactional memory will abort if a conflict
        // is detected, however no abort is generated when re-locking from the
        // same thread.
        //
        // Since locking the same mutex twice will result in two aliasing &mut
        // references, we instead create the mutex with type
        // PTHREAD_MUTEX_NORMAL which is guaranteed to deadlock if we try to
        // re-lock it from the same thread, thus avoiding undefined behavior.
        unsafe {
            let mut attr = MaybeUninit::<libc::pthread_mutexattr_t>::uninit();
            cvt_nz(libc::pthread_mutexattr_init(attr.as_mut_ptr())).unwrap();
            let attr = PthreadMutexAttr(&mut attr);
            cvt_nz(libc::pthread_mutexattr_settype(
                attr.0.as_mut_ptr(),
                libc::PTHREAD_MUTEX_NORMAL,
            ))
            .unwrap();
            cvt_nz(libc::pthread_mutex_init(mutex.0.get(), attr.0.as_ptr())).unwrap();
        }

        mutex
    }
}

impl Drop for AllocatedMutex {
    #[inline]
    fn drop(&mut self) {
        let r = unsafe { libc::pthread_mutex_destroy(self.0.get()) };
        if cfg!(target_os = "dragonfly") {
            // On DragonFly pthread_mutex_destroy() returns EINVAL if called on a
            // mutex that was just initialized with libc::PTHREAD_MUTEX_INITIALIZER.
            // Once it is used (locked/unlocked) or pthread_mutex_init() is called,
            // this behavior no longer occurs.
            debug_assert!(r == 0 || r == libc::EINVAL);
        } else {
            debug_assert_eq!(r, 0);
        }
    }
}

impl Mutex {
    #[inline]
    pub const fn new() -> Mutex {
        Mutex { inner: OnceBox::new() }
    }

    /// Gets access to the pthread mutex under the assumption that the mutex is
    /// locked.
    ///
    /// This allows skipping the initialization check, as the mutex can only be
    /// locked if it is already initialized, and allows relaxing the ordering
    /// on the pointer load, since the allocation cannot have been modified
    /// since the `lock` and the lock must have occurred on the current thread.
    ///
    /// # Safety
    /// Causes undefined behavior if the mutex is not locked.
    #[inline]
    pub(crate) unsafe fn get_assert_locked(&self) -> *mut libc::pthread_mutex_t {
        unsafe { self.inner.get_unchecked().0.get() }
    }

    #[inline]
    fn get(&self) -> *mut libc::pthread_mutex_t {
        // If initialization fails, the mutex is destroyed. This is always sound,
        // however, as the mutex cannot have been locked yet.
        self.inner.get_or_init(AllocatedMutex::new).0.get()
    }

    #[inline]
    pub fn lock(&self) {
        #[cold]
        #[inline(never)]
        fn fail(r: i32) -> ! {
            let error = Error::from_raw_os_error(r);
            panic!("failed to lock mutex: {error}");
        }

        let r = unsafe { libc::pthread_mutex_lock(self.get()) };
        // As we set the mutex type to `PTHREAD_MUTEX_NORMAL` above, we expect
        // the lock call to never fail. Unfortunately however, some platforms
        // (Solaris) do not conform to the standard, and instead always provide
        // deadlock detection. How kind of them! Unfortunately that means that
        // we need to check the error code here. To save us from UB on other
        // less well-behaved platforms in the future, we do it even on "good"
        // platforms like macOS. See #120147 for more context.
        if r != 0 {
            fail(r)
        }
    }

    #[inline]
    pub unsafe fn unlock(&self) {
        let r = libc::pthread_mutex_unlock(self.get_assert_locked());
        debug_assert_eq!(r, 0);
    }

    #[inline]
    pub fn try_lock(&self) -> bool {
        unsafe { libc::pthread_mutex_trylock(self.get()) == 0 }
    }
}

impl Drop for Mutex {
    fn drop(&mut self) {
        let Some(mutex) = self.inner.take() else { return };
        // We're not allowed to pthread_mutex_destroy a locked mutex,
        // so check first if it's unlocked.
        if unsafe { libc::pthread_mutex_trylock(mutex.0.get()) == 0 } {
            unsafe { libc::pthread_mutex_unlock(mutex.0.get()) };
            drop(mutex);
        } else {
            // The mutex is locked. This happens if a MutexGuard is leaked.
            // In this case, we just leak the Mutex too.
            forget(mutex);
        }
    }
}

pub(super) struct PthreadMutexAttr<'a>(pub &'a mut MaybeUninit<libc::pthread_mutexattr_t>);

impl Drop for PthreadMutexAttr<'_> {
    fn drop(&mut self) {
        unsafe {
            let result = libc::pthread_mutexattr_destroy(self.0.as_mut_ptr());
            debug_assert_eq!(result, 0);
        }
    }
}
