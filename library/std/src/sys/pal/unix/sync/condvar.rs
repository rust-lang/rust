use super::Mutex;
use crate::cell::UnsafeCell;
use crate::pin::Pin;
#[cfg(not(target_os = "nto"))]
use crate::sys::pal::time::TIMESPEC_MAX;
#[cfg(target_os = "nto")]
use crate::sys::pal::time::TIMESPEC_MAX_CAPPED;
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
}

#[cfg(not(target_vendor = "apple"))]
impl Condvar {
    /// # Safety
    /// * `init` must have been called on this instance.
    /// * `mutex` must be locked by the current thread.
    /// * This condition variable may only be used with the same mutex.
    pub unsafe fn wait_timeout(&self, mutex: Pin<&Mutex>, dur: Duration) -> bool {
        use crate::sys::pal::time::Timespec;

        let mutex = mutex.raw();

        // Cygwin's implementation is based on the NT API, which measures time
        // in units of 100 ns. Unfortunately, Cygwin does not properly guard
        // against overflow when converting the time, hence we clamp the interval
        // to 1000 years, which will only become a problem in around 27000 years,
        // when the next rollover is less than 1000 years away...
        #[cfg(target_os = "cygwin")]
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

// Apple platforms (since macOS version 10.4 and iOS version 2.0) have
// `pthread_cond_timedwait_relative_np`, a non-standard extension that
// measures timeouts based on the monotonic clock and is thus resilient
// against wall-clock changes.
#[cfg(target_vendor = "apple")]
impl Condvar {
    /// # Safety
    /// * `init` must have been called on this instance.
    /// * `mutex` must be locked by the current thread.
    /// * This condition variable may only be used with the same mutex.
    pub unsafe fn wait_timeout(&self, mutex: Pin<&Mutex>, dur: Duration) -> bool {
        let mutex = mutex.raw();

        // The macOS implementation of `pthread_cond_timedwait` internally
        // converts the timeout passed to `pthread_cond_timedwait_relative_np`
        // to nanoseconds. Unfortunately, the "psynch" variant of condvars does
        // not guard against overflow during the conversion[^1], which means
        // that `pthread_cond_timedwait_relative_np` will return `ETIMEDOUT`
        // much earlier than expected if the relative timeout is longer than
        // `u64::MAX` nanoseconds.
        //
        // This can be observed even on newer platforms (by setting the environment
        // variable PTHREAD_MUTEX_USE_ULOCK to a value other than "1") by calling e.g.
        // ```
        // condvar.wait_timeout(..., Duration::from_secs(u64::MAX.div_ceil(1_000_000_000));
        // ```
        // (see #37440, especially
        // https://github.com/rust-lang/rust/issues/37440#issuecomment-3285958326).
        //
        // To work around this issue, always clamp the timeout to u64::MAX nanoseconds,
        // even if the "ulock" variant is used (which does guard against overflow).
        //
        // [^1]: https://github.com/apple-oss-distributions/libpthread/blob/1ebf56b3a702df53213c2996e5e128a535d2577e/kern/kern_synch.c#L1269
        const MAX_DURATION: Duration = Duration::from_nanos(u64::MAX);

        let (dur, clamped) = if dur <= MAX_DURATION { (dur, false) } else { (MAX_DURATION, true) };

        // This can overflow on 32-bit platforms, but not on 64-bit because of the clamping above.
        let timeout = if let Ok(tv_sec) = dur.as_secs().try_into() {
            libc::timespec { tv_sec, tv_nsec: dur.subsec_nanos() as _ }
        } else {
            // This is less than `MAX_DURATION` on 32-bit platforms.
            TIMESPEC_MAX
        };

        let r = unsafe { libc::pthread_cond_timedwait_relative_np(self.raw(), mutex, &timeout) };
        assert!(r == libc::ETIMEDOUT || r == 0);
        // Report clamping as a spurious wakeup. Who knows, maybe some
        // interstellar space probe will rely on this ;-).
        r == 0 || clamped
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

#[cfg(target_vendor = "apple")]
impl Condvar {
    // `pthread_cond_timedwait_relative_np` measures the timeout
    // based on the monotonic clock.
    pub const PRECISE_TIMEOUT: bool = true;

    /// # Safety
    /// May only be called once per instance of `Self`.
    pub unsafe fn init(self: Pin<&mut Self>) {
        // `PTHREAD_COND_INITIALIZER` is fully supported and we don't need to
        // change clocks, so there's nothing to do here.
    }
}

// `pthread_condattr_setclock` is unfortunately not supported on these platforms.
#[cfg(any(
    target_os = "android",
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
