use super::Mutex;
use crate::cell::UnsafeCell;
use crate::pin::Pin;
#[cfg(not(target_os = "nto"))]
use crate::sys::pal::time::TIMESPEC_MAX;
#[cfg(target_os = "nto")]
use crate::sys::pal::time::TIMESPEC_MAX_CAPPED;
use crate::sys::pal::time::Timespec;
use crate::time::Duration;

pub struct Condvar {
    inner: UnsafeCell<libc::pthread_cond_t>,
}

impl Condvar {
    pub fn new() -> Condvar {
        Condvar { inner: UnsafeCell::new(libc::PTHREAD_COND_INITIALIZER) }
    }

    #[inline]
    fn raw(&self) -> *mut libc::pthread_cond_t {
        self.inner.get()
    }

    /// # Safety
    /// `init` must have been called on this instance.
    #[inline]
    pub unsafe fn notify_one(self: Pin<&Self>) {
        let r = unsafe { libc::pthread_cond_signal(self.raw()) };
        debug_assert_eq!(r, 0);
    }

    /// # Safety
    /// `init` must have been called on this instance.
    #[inline]
    pub unsafe fn notify_all(self: Pin<&Self>) {
        let r = unsafe { libc::pthread_cond_broadcast(self.raw()) };
        debug_assert_eq!(r, 0);
    }

    /// # Safety
    /// * `init` must have been called on this instance.
    /// * `mutex` must be locked by the current thread.
    /// * This condition variable may only be used with the same mutex.
    #[inline]
    pub unsafe fn wait(self: Pin<&Self>, mutex: Pin<&Mutex>) {
        let r = unsafe { libc::pthread_cond_wait(self.raw(), mutex.raw()) };
        debug_assert_eq!(r, 0);
    }

    /// # Safety
    /// * `init` must have been called on this instance.
    /// * `mutex` must be locked by the current thread.
    /// * This condition variable may only be used with the same mutex.
    pub unsafe fn wait_timeout(&self, mutex: Pin<&Mutex>, dur: Duration) -> bool {
        let mutex = mutex.raw();

        // OSX implementation of `pthread_cond_timedwait` is buggy
        // with super long durations. When duration is greater than
        // 0x100_0000_0000_0000 seconds, `pthread_cond_timedwait`
        // in macOS Sierra returns error 316.
        //
        // This program demonstrates the issue:
        // https://gist.github.com/stepancheg/198db4623a20aad2ad7cddb8fda4a63c
        //
        // To work around this issue, the timeout is clamped to 1000 years.
        //
        // Cygwin implementation is based on NT API and a super large timeout
        // makes the syscall block forever.
        #[cfg(any(target_vendor = "apple", target_os = "cygwin"))]
        let dur = Duration::min(dur, Duration::from_secs(1000 * 365 * 86400));

        let timeout = Timespec::now(Self::CLOCK).checked_add_duration(&dur);

        #[cfg(not(target_os = "nto"))]
        let timeout = timeout.and_then(|t| t.to_timespec()).unwrap_or(TIMESPEC_MAX);

        #[cfg(target_os = "nto")]
        let timeout = timeout.and_then(|t| t.to_timespec_capped()).unwrap_or(TIMESPEC_MAX_CAPPED);

        let r = unsafe { libc::pthread_cond_timedwait(self.raw(), mutex, &timeout) };
        assert!(r == libc::ETIMEDOUT || r == 0);
        r == 0
    }
}

#[cfg(not(any(
    target_os = "android",
    target_vendor = "apple",
    target_os = "espidf",
    target_os = "horizon",
    target_os = "l4re",
    target_os = "redox",
    target_os = "teeos",
)))]
impl Condvar {
    pub const PRECISE_TIMEOUT: bool = true;
    const CLOCK: libc::clockid_t = libc::CLOCK_MONOTONIC;

    /// # Safety
    /// May only be called once per instance of `Self`.
    pub unsafe fn init(self: Pin<&mut Self>) {
        use crate::mem::MaybeUninit;

        struct AttrGuard<'a>(pub &'a mut MaybeUninit<libc::pthread_condattr_t>);
        impl Drop for AttrGuard<'_> {
            fn drop(&mut self) {
                unsafe {
                    let result = libc::pthread_condattr_destroy(self.0.as_mut_ptr());
                    assert_eq!(result, 0);
                }
            }
        }

        unsafe {
            let mut attr = MaybeUninit::<libc::pthread_condattr_t>::uninit();
            let r = libc::pthread_condattr_init(attr.as_mut_ptr());
            assert_eq!(r, 0);
            let attr = AttrGuard(&mut attr);
            let r = libc::pthread_condattr_setclock(attr.0.as_mut_ptr(), Self::CLOCK);
            assert_eq!(r, 0);
            let r = libc::pthread_cond_init(self.raw(), attr.0.as_ptr());
            assert_eq!(r, 0);
        }
    }
}

// `pthread_condattr_setclock` is unfortunately not supported on these platforms.
#[cfg(any(
    target_os = "android",
    target_vendor = "apple",
    target_os = "espidf",
    target_os = "horizon",
    target_os = "l4re",
    target_os = "redox",
    target_os = "teeos",
))]
impl Condvar {
    pub const PRECISE_TIMEOUT: bool = false;
    const CLOCK: libc::clockid_t = libc::CLOCK_REALTIME;

    /// # Safety
    /// May only be called once per instance of `Self`.
    pub unsafe fn init(self: Pin<&mut Self>) {
        if cfg!(any(target_os = "espidf", target_os = "horizon", target_os = "teeos")) {
            // NOTE: ESP-IDF's PTHREAD_COND_INITIALIZER support is not released yet
            // So on that platform, init() should always be called.
            //
            // Similar story for the 3DS (horizon) and for TEEOS.
            let r = unsafe { libc::pthread_cond_init(self.raw(), crate::ptr::null()) };
            assert_eq!(r, 0);
        }
    }
}

impl !Unpin for Condvar {}

unsafe impl Sync for Condvar {}
unsafe impl Send for Condvar {}

impl Drop for Condvar {
    #[inline]
    fn drop(&mut self) {
        let r = unsafe { libc::pthread_cond_destroy(self.raw()) };
        if cfg!(target_os = "dragonfly") {
            // On DragonFly pthread_cond_destroy() returns EINVAL if called on
            // a condvar that was just initialized with
            // libc::PTHREAD_COND_INITIALIZER. Once it is used or
            // pthread_cond_init() is called, this behaviour no longer occurs.
            debug_assert!(r == 0 || r == libc::EINVAL);
        } else {
            debug_assert_eq!(r, 0);
        }
    }
}
