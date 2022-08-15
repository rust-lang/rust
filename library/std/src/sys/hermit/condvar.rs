use crate::ffi::c_void;
use crate::mem::MaybeUninit;
use crate::ptr;
use crate::sync::atomic::{
    AtomicPtr, AtomicUsize,
    Ordering::{Acquire, Release, SeqCst},
};
use crate::sys::hermit::abi;
use crate::sys::locks::Mutex;
use crate::time::Duration;

// The implementation is inspired by Andrew D. Birrell's paper
// "Implementing Condition Variables with Semaphores"

pub struct Condvar {
    counter: AtomicUsize,
    sem1: AtomicPtr<c_void>,
    sem2: AtomicPtr<c_void>,
}

pub(crate) type MovableCondvar = Condvar;

#[cold]
fn init_semaphore(sem: &AtomicPtr<c_void>) -> *mut c_void {
    let new = unsafe {
        let mut new = MaybeUninit::uninit();
        let _ = abi::sem_init(new.as_mut_ptr(), 0);
        new.assume_init() as *mut c_void
    };

    match sem.compare_exchange(ptr::null_mut(), new, Release, Acquire) {
        Ok(_) => new,
        Err(sem) => unsafe {
            let _ = abi::sem_destroy(new);
            sem
        },
    }
}

impl Condvar {
    #[inline]
    pub const fn new() -> Self {
        Self {
            counter: AtomicUsize::new(0),
            sem1: AtomicPtr::new(ptr::null_mut()),
            sem2: AtomicPtr::new(ptr::null_mut()),
        }
    }

    #[inline]
    fn semaphores(&self) -> (*const c_void, *const c_void) {
        let mut sem1 = self.sem1.load(Acquire);
        if sem1.is_null() {
            sem1 = init_semaphore(&self.sem1);
        }

        let mut sem2 = self.sem2.load(Acquire);
        if sem2.is_null() {
            sem2 = init_semaphore(&self.sem2);
        }

        (sem1, sem2)
    }

    pub unsafe fn notify_one(&self) {
        if self.counter.load(SeqCst) > 0 {
            self.counter.fetch_sub(1, SeqCst);
            let (sem1, sem2) = self.semaphores();
            unsafe {
                abi::sem_post(sem1);
                abi::sem_timedwait(sem2, 0);
            }
        }
    }

    pub unsafe fn notify_all(&self) {
        let counter = self.counter.swap(0, SeqCst);
        let (sem1, sem2) = self.semaphores();
        for _ in 0..counter {
            unsafe { abi::sem_post(sem1) };
        }
        for _ in 0..counter {
            unsafe { abi::sem_timedwait(sem2, 0) };
        }
    }

    pub unsafe fn wait(&self, mutex: &Mutex) {
        self.counter.fetch_add(1, SeqCst);
        let (sem1, sem2) = self.semaphores();
        mutex.unlock();
        abi::sem_timedwait(sem1, 0);
        abi::sem_post(sem2);
        mutex.lock();
    }

    pub unsafe fn wait_timeout(&self, mutex: &Mutex, dur: Duration) -> bool {
        self.counter.fetch_add(1, SeqCst);
        let (sem1, sem2) = self.semaphores();
        mutex.unlock();

        let millis = dur.as_millis().min(u32::MAX as u128) as u32;
        let res =
            if millis > 0 { abi::sem_timedwait(sem1, millis) } else { abi::sem_trywait(sem1) };

        abi::sem_post(sem2);
        mutex.lock();
        res == 0
    }
}

impl Drop for Condvar {
    fn drop(&mut self) {
        unsafe {
            let sem1 = *self.sem1.get_mut();
            let sem2 = *self.sem2.get_mut();
            if !sem1.is_null() {
                let _ = abi::sem_destroy(sem1);
            }
            if !sem2.is_null() {
                let _ = abi::sem_destroy(sem2);
            }
        }
    }
}
