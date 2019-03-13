use crate::cell::UnsafeCell;
use crate::intrinsics::{atomic_cxchg, atomic_load, atomic_xadd, atomic_xchg};
use crate::ptr;
use crate::time::Duration;

use crate::sys::mutex::{mutex_unlock, Mutex};
use crate::sys::syscall::{futex, TimeSpec, FUTEX_WAIT, FUTEX_WAKE, FUTEX_REQUEUE};

pub struct Condvar {
    lock: UnsafeCell<*mut i32>,
    seq: UnsafeCell<i32>
}

impl Condvar {
    pub const fn new() -> Condvar {
        Condvar {
            lock: UnsafeCell::new(ptr::null_mut()),
            seq: UnsafeCell::new(0)
        }
    }

    #[inline]
    pub unsafe fn init(&self) {
        *self.lock.get() = ptr::null_mut();
        *self.seq.get() = 0;
    }

    #[inline]
    pub fn notify_one(&self) {
        unsafe {
            let seq = self.seq.get();

            atomic_xadd(seq, 1);

            let _ = futex(seq, FUTEX_WAKE, 1, 0, ptr::null_mut());
        }
    }

    #[inline]
    pub fn notify_all(&self) {
        unsafe {
            let lock = self.lock.get();
            let seq = self.seq.get();

            if *lock == ptr::null_mut() {
                return;
            }

            atomic_xadd(seq, 1);

            let _ = futex(seq, FUTEX_REQUEUE, 1, crate::usize::MAX, *lock);
        }
    }

    #[inline]
    unsafe fn wait_inner(&self, mutex: &Mutex, timeout_ptr: *const TimeSpec) -> bool {
        let lock = self.lock.get();
        let seq = self.seq.get();

        if *lock != mutex.lock.get() {
            if *lock != ptr::null_mut() {
                panic!("Condvar used with more than one Mutex");
            }

            atomic_cxchg(lock as *mut usize, 0, mutex.lock.get() as usize);
        }

        mutex_unlock(*lock);

        let seq_before = atomic_load(seq);

        let _ = futex(seq, FUTEX_WAIT, seq_before, timeout_ptr as usize, ptr::null_mut());

        let seq_after = atomic_load(seq);

        while atomic_xchg(*lock, 2) != 0 {
            let _ = futex(*lock, FUTEX_WAIT, 2, 0, ptr::null_mut());
        }

        seq_before != seq_after
    }

    #[inline]
    pub fn wait(&self, mutex: &Mutex) {
        unsafe {
            assert!(self.wait_inner(mutex, ptr::null()));
        }
    }

    #[inline]
    pub fn wait_timeout(&self, mutex: &Mutex, dur: Duration) -> bool {
        unsafe {
            let timeout = TimeSpec {
                tv_sec: dur.as_secs() as i64,
                tv_nsec: dur.subsec_nanos() as i32
            };

            self.wait_inner(mutex, &timeout as *const TimeSpec)
        }
    }

    #[inline]
    pub unsafe fn destroy(&self) {
        *self.lock.get() = ptr::null_mut();
        *self.seq.get() = 0;
    }
}

unsafe impl Send for Condvar {}

unsafe impl Sync for Condvar {}
