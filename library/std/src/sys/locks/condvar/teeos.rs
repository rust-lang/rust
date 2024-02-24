use crate::cell::UnsafeCell;
use crate::ptr;
use crate::sync::atomic::{AtomicPtr, Ordering::Relaxed};
use crate::sys::locks::mutex::{self, Mutex};
use crate::sys::time::TIMESPEC_MAX;
use crate::sys_common::lazy_box::{LazyBox, LazyInit};
use crate::time::Duration;

extern "C" {
    pub fn pthread_cond_timedwait(
        cond: *mut libc::pthread_cond_t,
        lock: *mut libc::pthread_mutex_t,
        adstime: *const libc::timespec,
    ) -> libc::c_int;
}

struct AllocatedCondvar(UnsafeCell<libc::pthread_cond_t>);

pub struct Condvar {
    inner: LazyBox<AllocatedCondvar>,
    mutex: AtomicPtr<libc::pthread_mutex_t>,
}

#[inline]
fn raw(c: &Condvar) -> *mut libc::pthread_cond_t {
    c.inner.0.get()
}

unsafe impl Send for AllocatedCondvar {}
unsafe impl Sync for AllocatedCondvar {}

impl LazyInit for AllocatedCondvar {
    fn init() -> Box<Self> {
        let condvar = Box::new(AllocatedCondvar(UnsafeCell::new(libc::PTHREAD_COND_INITIALIZER)));

        let r = unsafe { libc::pthread_cond_init(condvar.0.get(), crate::ptr::null()) };
        assert_eq!(r, 0);

        condvar
    }
}

impl Drop for AllocatedCondvar {
    #[inline]
    fn drop(&mut self) {
        let r = unsafe { libc::pthread_cond_destroy(self.0.get()) };
        debug_assert_eq!(r, 0);
    }
}

impl Condvar {
    pub const fn new() -> Condvar {
        Condvar { inner: LazyBox::new(), mutex: AtomicPtr::new(ptr::null_mut()) }
    }

    #[inline]
    fn verify(&self, mutex: *mut libc::pthread_mutex_t) {
        match self.mutex.compare_exchange(ptr::null_mut(), mutex, Relaxed, Relaxed) {
            Ok(_) => {}                // Stored the address
            Err(n) if n == mutex => {} // Lost a race to store the same address
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
        let mutex = mutex::raw(mutex);
        self.verify(mutex);
        let r = libc::pthread_cond_wait(raw(self), mutex);
        debug_assert_eq!(r, 0);
    }

    pub unsafe fn wait_timeout(&self, mutex: &Mutex, dur: Duration) -> bool {
        use crate::sys::time::Timespec;

        let mutex = mutex::raw(mutex);
        self.verify(mutex);

        let timeout = Timespec::now(libc::CLOCK_MONOTONIC)
            .checked_add_duration(&dur)
            .and_then(|t| t.to_timespec())
            .unwrap_or(TIMESPEC_MAX);

        let r = pthread_cond_timedwait(raw(self), mutex, &timeout);
        assert!(r == libc::ETIMEDOUT || r == 0);
        r == 0
    }
}
