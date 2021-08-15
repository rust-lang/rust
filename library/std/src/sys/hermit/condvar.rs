use crate::ffi::c_void;
use crate::ptr;
use crate::sync::atomic::{AtomicUsize, Ordering::SeqCst};
use crate::sys::hermit::abi;
use crate::sys::mutex::Mutex;
use crate::time::Duration;

// The implementation is inspired by Andrew D. Birrell's paper
// "Implementing Condition Variables with Semaphores"

pub struct Condvar {
    counter: AtomicUsize,
    sem1: *const c_void,
    sem2: *const c_void,
}

pub type MovableCondvar = Condvar;

unsafe impl Send for Condvar {}
unsafe impl Sync for Condvar {}

impl Condvar {
    pub const fn new() -> Condvar {
        Condvar { counter: AtomicUsize::new(0), sem1: ptr::null(), sem2: ptr::null() }
    }

    pub unsafe fn init(&mut self) {
        let _ = abi::sem_init(&mut self.sem1 as *mut *const c_void, 0);
        let _ = abi::sem_init(&mut self.sem2 as *mut *const c_void, 0);
    }

    pub unsafe fn notify_one(&self) {
        if self.counter.load(SeqCst) > 0 {
            self.counter.fetch_sub(1, SeqCst);
            abi::sem_post(self.sem1);
            abi::sem_timedwait(self.sem2, 0);
        }
    }

    pub unsafe fn notify_all(&self) {
        let counter = self.counter.swap(0, SeqCst);
        for _ in 0..counter {
            abi::sem_post(self.sem1);
        }
        for _ in 0..counter {
            abi::sem_timedwait(self.sem2, 0);
        }
    }

    pub unsafe fn wait(&self, mutex: &Mutex) {
        self.counter.fetch_add(1, SeqCst);
        mutex.unlock();
        abi::sem_timedwait(self.sem1, 0);
        abi::sem_post(self.sem2);
        mutex.lock();
    }

    pub unsafe fn wait_timeout(&self, _mutex: &Mutex, _dur: Duration) -> bool {
        panic!("wait_timeout not supported on hermit");
    }

    pub unsafe fn destroy(&self) {
        let _ = abi::sem_destroy(self.sem1);
        let _ = abi::sem_destroy(self.sem2);
    }
}
