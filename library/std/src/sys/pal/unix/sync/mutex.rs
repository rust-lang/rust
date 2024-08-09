use super::super::cvt_nz;
use crate::cell::UnsafeCell;
use crate::io::Error;
use crate::mem::MaybeUninit;
use crate::pin::Pin;

pub struct Mutex {
    inner: UnsafeCell<libc::pthread_mutex_t>,
}

impl Mutex {
    pub fn new() -> Mutex {
        Mutex { inner: UnsafeCell::new(libc::PTHREAD_MUTEX_INITIALIZER) }
    }

    pub(super) fn raw(&self) -> *mut libc::pthread_mutex_t {
        self.inner.get()
    }

    /// # Safety
    /// Must only be called once.
    pub unsafe fn init(self: Pin<&mut Self>) {
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
            let attr = AttrGuard(&mut attr);
            cvt_nz(libc::pthread_mutexattr_settype(
                attr.0.as_mut_ptr(),
                libc::PTHREAD_MUTEX_NORMAL,
            ))
            .unwrap();
            cvt_nz(libc::pthread_mutex_init(self.raw(), attr.0.as_ptr())).unwrap();
        }
    }

    /// # Safety
    /// * If `init` was not called, reentrant locking causes undefined behaviour.
    /// * Destroying a locked mutex causes undefined behaviour.
    pub unsafe fn lock(self: Pin<&Self>) {
        #[cold]
        #[inline(never)]
        fn fail(r: i32) -> ! {
            let error = Error::from_raw_os_error(r);
            panic!("failed to lock mutex: {error}");
        }

        let r = unsafe { libc::pthread_mutex_lock(self.raw()) };
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

    /// # Safety
    /// * If `init` was not called, reentrant locking causes undefined behaviour.
    /// * Destroying a locked mutex causes undefined behaviour.
    pub unsafe fn try_lock(self: Pin<&Self>) -> bool {
        unsafe { libc::pthread_mutex_trylock(self.raw()) == 0 }
    }

    /// # Safety
    /// The mutex must be locked by the current thread.
    pub unsafe fn unlock(self: Pin<&Self>) {
        let r = unsafe { libc::pthread_mutex_unlock(self.raw()) };
        debug_assert_eq!(r, 0);
    }
}

impl !Unpin for Mutex {}

unsafe impl Send for Mutex {}
unsafe impl Sync for Mutex {}

impl Drop for Mutex {
    fn drop(&mut self) {
        // SAFETY:
        // If `lock` or `init` was called, the mutex must have been pinned, so
        // it is still at the same location. Otherwise, `inner` must contain
        // `PTHREAD_MUTEX_INITIALIZER`, which is valid at all locations. Thus,
        // this call always destroys a valid mutex.
        let r = unsafe { libc::pthread_mutex_destroy(self.raw()) };
        if cfg!(target_os = "dragonfly") {
            // On DragonFly pthread_mutex_destroy() returns EINVAL if called on a
            // mutex that was just initialized with libc::PTHREAD_MUTEX_INITIALIZER.
            // Once it is used (locked/unlocked) or pthread_mutex_init() is called,
            // this behaviour no longer occurs.
            debug_assert!(r == 0 || r == libc::EINVAL);
        } else {
            debug_assert_eq!(r, 0);
        }
    }
}

struct AttrGuard<'a>(pub &'a mut MaybeUninit<libc::pthread_mutexattr_t>);

impl Drop for AttrGuard<'_> {
    fn drop(&mut self) {
        unsafe {
            let result = libc::pthread_mutexattr_destroy(self.0.as_mut_ptr());
            assert_eq!(result, 0);
        }
    }
}
