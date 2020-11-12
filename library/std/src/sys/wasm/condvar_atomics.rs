use crate::arch::wasm32;
use crate::cmp;
use crate::mem;
use crate::sync::atomic::{AtomicUsize, Ordering::SeqCst};
use crate::sys::mutex::Mutex;
use crate::time::Duration;

pub struct Condvar {
    cnt: AtomicUsize,
}

pub type MovableCondvar = Condvar;

// Condition variables are implemented with a simple counter internally that is
// likely to cause spurious wakeups. Blocking on a condition variable will first
// read the value of the internal counter, unlock the given mutex, and then
// block if and only if the counter's value is still the same. Notifying a
// condition variable will modify the counter (add one for now) and then wake up
// a thread waiting on the address of the counter.
//
// A thread waiting on the condition variable will as a result avoid going to
// sleep if it's notified after the lock is unlocked but before it fully goes to
// sleep. A sleeping thread is guaranteed to be woken up at some point as it can
// only be woken up with a call to `wake`.
//
// Note that it's possible for 2 or more threads to be woken up by a call to
// `notify_one` with this implementation. That can happen where the modification
// of `cnt` causes any threads in the middle of `wait` to avoid going to sleep,
// and the subsequent `wake` may wake up a thread that's actually blocking. We
// consider this a spurious wakeup, though, which all users of condition
// variables must already be prepared to handle. As a result, this source of
// spurious wakeups is currently though to be ok, although it may be problematic
// later on if it causes too many spurious wakeups.

impl Condvar {
    pub const fn new() -> Condvar {
        Condvar { cnt: AtomicUsize::new(0) }
    }

    #[inline]
    pub unsafe fn init(&mut self) {
        // nothing to do
    }

    pub unsafe fn notify_one(&self) {
        self.cnt.fetch_add(1, SeqCst);
        // SAFETY: ptr() is always valid
        unsafe {
            wasm32::memory_atomic_notify(self.ptr(), 1);
        }
    }

    #[inline]
    pub unsafe fn notify_all(&self) {
        self.cnt.fetch_add(1, SeqCst);
        // SAFETY: ptr() is always valid
        unsafe {
            wasm32::memory_atomic_notify(self.ptr(), u32::MAX); // -1 == "wake everyone"
        }
    }

    pub unsafe fn wait(&self, mutex: &Mutex) {
        // "atomically block and unlock" implemented by loading our current
        // counter's value, unlocking the mutex, and blocking if the counter
        // still has the same value.
        //
        // Notifications happen by incrementing the counter and then waking a
        // thread. Incrementing the counter after we unlock the mutex will
        // prevent us from sleeping and otherwise the call to `wake` will
        // wake us up once we're asleep.
        let ticket = self.cnt.load(SeqCst) as i32;
        mutex.unlock();
        let val = wasm32::memory_atomic_wait32(self.ptr(), ticket, -1);
        // 0 == woken, 1 == not equal to `ticket`, 2 == timeout (shouldn't happen)
        debug_assert!(val == 0 || val == 1);
        mutex.lock();
    }

    pub unsafe fn wait_timeout(&self, mutex: &Mutex, dur: Duration) -> bool {
        let ticket = self.cnt.load(SeqCst) as i32;
        mutex.unlock();
        let nanos = dur.as_nanos();
        let nanos = cmp::min(i64::MAX as u128, nanos);

        // If the return value is 2 then a timeout happened, so we return
        // `false` as we weren't actually notified.
        let ret = wasm32::memory_atomic_wait32(self.ptr(), ticket, nanos as i64) != 2;
        mutex.lock();
        return ret;
    }

    #[inline]
    pub unsafe fn destroy(&self) {
        // nothing to do
    }

    #[inline]
    fn ptr(&self) -> *mut i32 {
        assert_eq!(mem::size_of::<usize>(), mem::size_of::<i32>());
        self.cnt.as_mut_ptr() as *mut i32
    }
}
