use crate::cell::UnsafeCell;
use crate::sys::mutex::{self, Mutex};
use crate::time::Duration;

pub struct Condvar {
    inner: UnsafeCell<libc::pthread_cond_t>,
}

unsafe impl Send for Condvar {}
unsafe impl Sync for Condvar {}

const TIMESPEC_MAX: libc::timespec =
    libc::timespec { tv_sec: <libc::time_t>::MAX, tv_nsec: 1_000_000_000 - 1 };

fn saturating_cast_to_time_t(value: u64) -> libc::time_t {
    if value > <libc::time_t>::MAX as u64 { <libc::time_t>::MAX } else { value as libc::time_t }
}

impl Condvar {
    pub const fn new() -> Condvar {
        // Might be moved and address is changing it is better to avoid
        // initialization of potentially opaque OS data before it landed
        Condvar { inner: UnsafeCell::new(libc::PTHREAD_COND_INITIALIZER) }
    }

    pub unsafe fn init(&mut self) {
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
    pub unsafe fn notify_one(&self) {
        let r = libc::pthread_cond_signal(self.inner.get());
        debug_assert_eq!(r, 0);
    }

    #[inline]
    pub unsafe fn notify_all(&self) {
        let r = libc::pthread_cond_broadcast(self.inner.get());
        debug_assert_eq!(r, 0);
    }

    #[inline]
    pub unsafe fn wait(&self, mutex: &Mutex) {
        let r = libc::pthread_cond_wait(self.inner.get(), mutex::raw(mutex));
        debug_assert_eq!(r, 0);
    }

    // This implementation is used on systems that support pthread_condattr_setclock
    // where we configure condition variable to use monotonic clock (instead of
    // default system clock). This approach avoids all problems that result
    // from changes made to the system time.
    pub unsafe fn wait_timeout(&self, mutex: &Mutex, dur: Duration) -> bool {
        use crate::mem;

        let mut now: libc::timespec = mem::zeroed();
        let r = libc::clock_gettime(libc::CLOCK_MONOTONIC, &mut now);
        assert_eq!(r, 0);

        // Nanosecond calculations can't overflow because both values are below 1e9.
        let nsec = dur.subsec_nanos() + now.tv_nsec as u32;

        let sec = saturating_cast_to_time_t(dur.as_secs())
            .checked_add((nsec / 1_000_000_000) as libc::time_t)
            .and_then(|s| s.checked_add(now.tv_sec));
        let nsec = nsec % 1_000_000_000;

        let timeout =
            sec.map(|s| libc::timespec { tv_sec: s, tv_nsec: nsec as _ }).unwrap_or(TIMESPEC_MAX);

        let r = libc::pthread_cond_timedwait(self.inner.get(), mutex::raw(mutex), &timeout);
        assert!(r == libc::ETIMEDOUT || r == 0);
        r == 0
    }

    #[inline]
    pub unsafe fn destroy(&self) {
        let r = libc::pthread_cond_destroy(self.inner.get());
        debug_assert_eq!(r, 0);
    }
}
