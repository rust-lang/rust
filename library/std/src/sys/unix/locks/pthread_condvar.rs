use crate::cell::UnsafeCell;
use crate::ptr;
use crate::sync::atomic::{AtomicPtr, Ordering::Relaxed};
use crate::sys::locks::{pthread_mutex, Mutex};
use crate::sys_common::lazy_box::{LazyBox, LazyInit};
use crate::time::Duration;

const TIMESPEC_MAX: libc::timespec =
    libc::timespec { tv_sec: <libc::time_t>::MAX, tv_nsec: 1_000_000_000 - 1 };

pub struct Condvar {
    boxed: LazyBox<StaticCondvar>,
}

#[inline]
fn raw(condvar: &Condvar) -> *mut libc::pthread_cond_t {
    condvar.boxed.inner.get()
}

impl Condvar {
    #[inline]
    pub const fn new() -> Condvar {
        Condvar { boxed: LazyBox::new() }
    }

    #[inline]
    fn verify(&self, mutex: &Mutex) {
        let addr = pthread_mutex::raw(mutex);
        match self.boxed.mutex_addr.compare_exchange(ptr::null_mut(), addr, Relaxed, Relaxed) {
            // Stored the address
            Ok(_) => {}
            // Lost a race to store the same address
            Err(n) if n == addr => {}
            _ => panic!("attempted to use a condition variable with two mutexes"),
        }
    }

    #[inline]
    pub fn notify_one(&self) {
        let r = unsafe { libc::pthread_cond_signal(raw(self)) };
        debug_assert_eq!(r, 0);
    }

    #[inline]
    pub fn notify_all(&self) {
        let r = unsafe { libc::pthread_cond_broadcast(raw(self)) };
        debug_assert_eq!(r, 0);
    }

    #[inline]
    pub unsafe fn wait(&self, mutex: &Mutex) {
        self.verify(mutex);
        let r = libc::pthread_cond_wait(raw(self), pthread_mutex::raw(mutex));
        debug_assert_eq!(r, 0);
    }

    pub unsafe fn wait_timeout(&self, mutex: &Mutex, dur: Duration) -> bool {
        // Use the system clock on systems that do not support pthread_condattr_setclock.
        // This unfortunately results in problems when the system time changes.
        #[cfg(any(
            target_os = "macos",
            target_os = "ios",
            target_os = "watchos",
            target_os = "espidf",
            target_os = "horizon",
        ))]
        let (now, dur) = {
            use crate::cmp::min;
            use crate::sys::time::SystemTime;

            // OSX implementation of `pthread_cond_timedwait` is buggy
            // with super long durations. When duration is greater than
            // 0x100_0000_0000_0000 seconds, `pthread_cond_timedwait`
            // in macOS Sierra return error 316.
            //
            // This program demonstrates the issue:
            // https://gist.github.com/stepancheg/198db4623a20aad2ad7cddb8fda4a63c
            //
            // To work around this issue, and possible bugs of other OSes, timeout
            // is clamped to 1000 years, which is allowable per the API of `park_timeout`
            // because of spurious wakeups.
            let dur = min(dur, Duration::from_secs(1000 * 365 * 86400));
            let now = SystemTime::now().t;
            (now, dur)
        };
        // Use the monotonic clock on other systems.
        #[cfg(not(any(
            target_os = "macos",
            target_os = "ios",
            target_os = "watchos",
            target_os = "espidf",
            target_os = "horizon",
        )))]
        let (now, dur) = {
            use crate::sys::time::Timespec;

            (Timespec::now(libc::CLOCK_MONOTONIC), dur)
        };

        let timeout =
            now.checked_add_duration(&dur).and_then(|t| t.to_timespec()).unwrap_or(TIMESPEC_MAX);
        self.verify(mutex);
        let r = libc::pthread_cond_timedwait(raw(self), pthread_mutex::raw(mutex), &timeout);
        // The mutex will be locked no matter the error, so only check it in debug mode.
        debug_assert!(r == libc::ETIMEDOUT || r == 0);
        r == 0
    }
}

struct StaticCondvar {
    inner: UnsafeCell<libc::pthread_cond_t>,
    mutex_addr: AtomicPtr<libc::pthread_mutex_t>,
}

unsafe impl Send for StaticCondvar {}
unsafe impl Sync for StaticCondvar {}

impl LazyInit for StaticCondvar {
    fn init() -> Box<Self> {
        let mut condvar = Box::new(StaticCondvar {
            inner: UnsafeCell::new(libc::PTHREAD_COND_INITIALIZER),
            mutex_addr: AtomicPtr::new(ptr::null_mut()),
        });
        unsafe { condvar.init() };
        condvar
    }
}

impl StaticCondvar {
    #[cfg(any(
        target_os = "macos",
        target_os = "ios",
        target_os = "watchos",
        target_os = "l4re",
        target_os = "android",
        target_os = "redox"
    ))]
    unsafe fn init(&mut self) {}

    // NOTE: ESP-IDF's PTHREAD_COND_INITIALIZER support is not released yet
    // So on that platform, init() should always be called
    // Moreover, that platform does not have pthread_condattr_setclock support,
    // hence that initialization should be skipped as well
    //
    // Similar story for the 3DS (horizon).
    #[cfg(any(target_os = "espidf", target_os = "horizon"))]
    unsafe fn init(&mut self) {
        let r = libc::pthread_cond_init(self.inner.get(), crate::ptr::null());
        assert_eq!(r, 0);
    }

    #[cfg(not(any(
        target_os = "macos",
        target_os = "ios",
        target_os = "watchos",
        target_os = "l4re",
        target_os = "android",
        target_os = "redox",
        target_os = "espidf",
        target_os = "horizon"
    )))]
    unsafe fn init(&mut self) {
        use crate::mem::MaybeUninit;
        let mut attr = MaybeUninit::<libc::pthread_condattr_t>::uninit();
        let r = libc::pthread_condattr_init(attr.as_mut_ptr());
        assert_eq!(r, 0);
        let r = libc::pthread_condattr_setclock(attr.as_mut_ptr(), libc::CLOCK_MONOTONIC);
        assert_eq!(r, 0);
        let r = libc::pthread_cond_init(self.inner.get(), attr.as_ptr());
        assert_eq!(r, 0);
        let r = libc::pthread_condattr_destroy(attr.as_mut_ptr());
        assert_eq!(r, 0);
    }

    #[inline]
    #[cfg(not(target_os = "dragonfly"))]
    unsafe fn destroy(&mut self) {
        let r = libc::pthread_cond_destroy(self.inner.get());
        debug_assert_eq!(r, 0);
    }

    #[inline]
    #[cfg(target_os = "dragonfly")]
    unsafe fn destroy(&mut self) {
        let r = libc::pthread_cond_destroy(self.inner.get());
        // On DragonFly pthread_cond_destroy() returns EINVAL if called on
        // a condvar that was just initialized with
        // libc::PTHREAD_COND_INITIALIZER. Once it is used or
        // pthread_cond_init() is called, this behaviour no longer occurs.
        debug_assert!(r == 0 || r == libc::EINVAL);
    }
}

impl Drop for StaticCondvar {
    #[inline]
    fn drop(&mut self) {
        unsafe { self.destroy() };
    }
}
