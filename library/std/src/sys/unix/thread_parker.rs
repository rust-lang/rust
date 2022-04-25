//! Thread parking without `futex` using the `pthread` synchronization primitives.

#![cfg(not(any(
    target_os = "linux",
    target_os = "android",
    all(target_os = "emscripten", target_feature = "atomics")
)))]

use crate::cell::UnsafeCell;
use crate::marker::PhantomPinned;
use crate::pin::Pin;
use crate::ptr::addr_of_mut;
use crate::sync::atomic::AtomicUsize;
use crate::sync::atomic::Ordering::SeqCst;
use crate::time::Duration;

const EMPTY: usize = 0;
const PARKED: usize = 1;
const NOTIFIED: usize = 2;

unsafe fn lock(lock: *mut libc::pthread_mutex_t) {
    let r = libc::pthread_mutex_lock(lock);
    debug_assert_eq!(r, 0);
}

unsafe fn unlock(lock: *mut libc::pthread_mutex_t) {
    let r = libc::pthread_mutex_unlock(lock);
    debug_assert_eq!(r, 0);
}

unsafe fn notify_one(cond: *mut libc::pthread_cond_t) {
    let r = libc::pthread_cond_signal(cond);
    debug_assert_eq!(r, 0);
}

unsafe fn wait(cond: *mut libc::pthread_cond_t, lock: *mut libc::pthread_mutex_t) {
    let r = libc::pthread_cond_wait(cond, lock);
    debug_assert_eq!(r, 0);
}

const TIMESPEC_MAX: libc::timespec =
    libc::timespec { tv_sec: <libc::time_t>::MAX, tv_nsec: 1_000_000_000 - 1 };

fn saturating_cast_to_time_t(value: u64) -> libc::time_t {
    if value > <libc::time_t>::MAX as u64 { <libc::time_t>::MAX } else { value as libc::time_t }
}

// This implementation is used on systems that support pthread_condattr_setclock
// where we configure the condition variable to use the monotonic clock (instead of
// the default system clock). This approach avoids all problems that result
// from changes made to the system time.
#[cfg(not(any(target_os = "macos", target_os = "ios", target_os = "espidf")))]
unsafe fn wait_timeout(
    cond: *mut libc::pthread_cond_t,
    lock: *mut libc::pthread_mutex_t,
    dur: Duration,
) {
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
    let r = libc::pthread_cond_timedwait(cond, lock, &timeout);
    assert!(r == libc::ETIMEDOUT || r == 0);
}

// This implementation is modeled after libcxx's condition_variable
// https://github.com/llvm-mirror/libcxx/blob/release_35/src/condition_variable.cpp#L46
// https://github.com/llvm-mirror/libcxx/blob/release_35/include/__mutex_base#L367
#[cfg(any(target_os = "macos", target_os = "ios", target_os = "espidf"))]
unsafe fn wait_timeout(
    cond: *mut libc::pthread_cond_t,
    lock: *mut libc::pthread_mutex_t,
    mut dur: Duration,
) {
    use crate::ptr;

    // 1000 years
    let max_dur = Duration::from_secs(1000 * 365 * 86400);

    if dur > max_dur {
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
        dur = max_dur;
    }

    let mut sys_now = libc::timeval { tv_sec: 0, tv_usec: 0 };
    let r = libc::gettimeofday(&mut sys_now, ptr::null_mut());
    debug_assert_eq!(r, 0);
    let nsec = dur.subsec_nanos() as libc::c_long + (sys_now.tv_usec * 1000) as libc::c_long;
    let extra = (nsec / 1_000_000_000) as libc::time_t;
    let nsec = nsec % 1_000_000_000;
    let seconds = saturating_cast_to_time_t(dur.as_secs());
    let timeout = sys_now
        .tv_sec
        .checked_add(extra)
        .and_then(|s| s.checked_add(seconds))
        .map(|s| libc::timespec { tv_sec: s, tv_nsec: nsec })
        .unwrap_or(TIMESPEC_MAX);
    // And wait!
    let r = libc::pthread_cond_timedwait(cond, lock, &timeout);
    debug_assert!(r == libc::ETIMEDOUT || r == 0);
}

pub struct Parker {
    state: AtomicUsize,
    lock: UnsafeCell<libc::pthread_mutex_t>,
    cvar: UnsafeCell<libc::pthread_cond_t>,
    // The `pthread` primitives require a stable address, so make this struct `!Unpin`.
    _pinned: PhantomPinned,
}

impl Parker {
    /// Construct the UNIX parker in-place.
    ///
    /// # Safety
    /// The constructed parker must never be moved.
    pub unsafe fn new(parker: *mut Parker) {
        // Use the default mutex implementation to allow for simpler initialization.
        // This could lead to undefined behaviour when deadlocking. This is avoided
        // by not deadlocking. Note in particular the unlocking operation before any
        // panic, as code after the panic could try to park again.
        addr_of_mut!((*parker).state).write(AtomicUsize::new(EMPTY));
        addr_of_mut!((*parker).lock).write(UnsafeCell::new(libc::PTHREAD_MUTEX_INITIALIZER));

        cfg_if::cfg_if! {
            if #[cfg(any(
                target_os = "macos",
                target_os = "ios",
                target_os = "l4re",
                target_os = "android",
                target_os = "redox"
            ))] {
                addr_of_mut!((*parker).cvar).write(UnsafeCell::new(libc::PTHREAD_COND_INITIALIZER));
            } else if #[cfg(target_os = "espidf")] {
                let r = libc::pthread_cond_init(addr_of_mut!((*parker).cvar).cast(), crate::ptr::null());
                assert_eq!(r, 0);
            } else {
                use crate::mem::MaybeUninit;
                let mut attr = MaybeUninit::<libc::pthread_condattr_t>::uninit();
                let r = libc::pthread_condattr_init(attr.as_mut_ptr());
                assert_eq!(r, 0);
                let r = libc::pthread_condattr_setclock(attr.as_mut_ptr(), libc::CLOCK_MONOTONIC);
                assert_eq!(r, 0);
                let r = libc::pthread_cond_init(addr_of_mut!((*parker).cvar).cast(), attr.as_ptr());
                assert_eq!(r, 0);
                let r = libc::pthread_condattr_destroy(attr.as_mut_ptr());
                assert_eq!(r, 0);
            }
        }
    }

    // This implementation doesn't require `unsafe`, but other implementations
    // may assume this is only called by the thread that owns the Parker.
    pub unsafe fn park(self: Pin<&Self>) {
        // If we were previously notified then we consume this notification and
        // return quickly.
        if self.state.compare_exchange(NOTIFIED, EMPTY, SeqCst, SeqCst).is_ok() {
            return;
        }

        // Otherwise we need to coordinate going to sleep
        lock(self.lock.get());
        match self.state.compare_exchange(EMPTY, PARKED, SeqCst, SeqCst) {
            Ok(_) => {}
            Err(NOTIFIED) => {
                // We must read here, even though we know it will be `NOTIFIED`.
                // This is because `unpark` may have been called again since we read
                // `NOTIFIED` in the `compare_exchange` above. We must perform an
                // acquire operation that synchronizes with that `unpark` to observe
                // any writes it made before the call to unpark. To do that we must
                // read from the write it made to `state`.
                let old = self.state.swap(EMPTY, SeqCst);

                unlock(self.lock.get());

                assert_eq!(old, NOTIFIED, "park state changed unexpectedly");
                return;
            } // should consume this notification, so prohibit spurious wakeups in next park.
            Err(_) => {
                unlock(self.lock.get());

                panic!("inconsistent park state")
            }
        }

        loop {
            wait(self.cvar.get(), self.lock.get());

            match self.state.compare_exchange(NOTIFIED, EMPTY, SeqCst, SeqCst) {
                Ok(_) => break, // got a notification
                Err(_) => {}    // spurious wakeup, go back to sleep
            }
        }

        unlock(self.lock.get());
    }

    // This implementation doesn't require `unsafe`, but other implementations
    // may assume this is only called by the thread that owns the Parker. Use
    // `Pin` to guarantee a stable address for the mutex and condition variable.
    pub unsafe fn park_timeout(self: Pin<&Self>, dur: Duration) {
        // Like `park` above we have a fast path for an already-notified thread, and
        // afterwards we start coordinating for a sleep.
        // return quickly.
        if self.state.compare_exchange(NOTIFIED, EMPTY, SeqCst, SeqCst).is_ok() {
            return;
        }

        lock(self.lock.get());
        match self.state.compare_exchange(EMPTY, PARKED, SeqCst, SeqCst) {
            Ok(_) => {}
            Err(NOTIFIED) => {
                // We must read again here, see `park`.
                let old = self.state.swap(EMPTY, SeqCst);
                unlock(self.lock.get());

                assert_eq!(old, NOTIFIED, "park state changed unexpectedly");
                return;
            } // should consume this notification, so prohibit spurious wakeups in next park.
            Err(_) => {
                unlock(self.lock.get());
                panic!("inconsistent park_timeout state")
            }
        }

        // Wait with a timeout, and if we spuriously wake up or otherwise wake up
        // from a notification we just want to unconditionally set the state back to
        // empty, either consuming a notification or un-flagging ourselves as
        // parked.
        wait_timeout(self.cvar.get(), self.lock.get(), dur);

        match self.state.swap(EMPTY, SeqCst) {
            NOTIFIED => unlock(self.lock.get()), // got a notification, hurray!
            PARKED => unlock(self.lock.get()),   // no notification, alas
            n => {
                unlock(self.lock.get());
                panic!("inconsistent park_timeout state: {n}")
            }
        }
    }

    pub fn unpark(self: Pin<&Self>) {
        // To ensure the unparked thread will observe any writes we made
        // before this call, we must perform a release operation that `park`
        // can synchronize with. To do that we must write `NOTIFIED` even if
        // `state` is already `NOTIFIED`. That is why this must be a swap
        // rather than a compare-and-swap that returns if it reads `NOTIFIED`
        // on failure.
        match self.state.swap(NOTIFIED, SeqCst) {
            EMPTY => return,    // no one was waiting
            NOTIFIED => return, // already unparked
            PARKED => {}        // gotta go wake someone up
            _ => panic!("inconsistent state in unpark"),
        }

        // There is a period between when the parked thread sets `state` to
        // `PARKED` (or last checked `state` in the case of a spurious wake
        // up) and when it actually waits on `cvar`. If we were to notify
        // during this period it would be ignored and then when the parked
        // thread went to sleep it would never wake up. Fortunately, it has
        // `lock` locked at this stage so we can acquire `lock` to wait until
        // it is ready to receive the notification.
        //
        // Releasing `lock` before the call to `notify_one` means that when the
        // parked thread wakes it doesn't get woken only to have to wait for us
        // to release `lock`.
        unsafe {
            lock(self.lock.get());
            unlock(self.lock.get());
            notify_one(self.cvar.get());
        }
    }
}

impl Drop for Parker {
    fn drop(&mut self) {
        unsafe {
            libc::pthread_cond_destroy(self.cvar.get_mut());
            libc::pthread_mutex_destroy(self.lock.get_mut());
        }
    }
}

unsafe impl Sync for Parker {}
unsafe impl Send for Parker {}
