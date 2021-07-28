use fortanix_sgx_abi::Tcs;

use super::abi::thread;

use super::waitqueue::{try_lock_or_false, NotifiedTcs, SpinMutex, WaitQueue, WaitVariable};

pub struct Mutex {
    inner: SpinMutex<WaitVariable<bool>>,
}

// not movable: see UnsafeList implementation
pub type MovableMutex = Box<Mutex>;

// Implementation according to “Operating Systems: Three Easy Pieces”, chapter 28
impl Mutex {
    pub const fn new() -> Mutex {
        Mutex { inner: SpinMutex::new(WaitVariable::new(false)) }
    }

    #[inline]
    pub unsafe fn init(&mut self) {}

    #[inline]
    pub unsafe fn lock(&self) {
        let mut guard = self.inner.lock();
        if *guard.lock_var() {
            // Another thread has the lock, wait
            WaitQueue::wait(guard, || {})
        // Another thread has passed the lock to us
        } else {
            // We are just now obtaining the lock
            *guard.lock_var_mut() = true;
        }
    }

    #[inline]
    pub unsafe fn unlock(&self) {
        let guard = self.inner.lock();
        if let Err(mut guard) = WaitQueue::notify_one(guard) {
            // No other waiters, unlock
            *guard.lock_var_mut() = false;
        } else {
            // There was a thread waiting, just pass the lock
        }
    }

    #[inline]
    pub unsafe fn try_lock(&self) -> bool {
        let mut guard = try_lock_or_false!(self.inner);
        if *guard.lock_var() {
            // Another thread has the lock
            false
        } else {
            // We are just now obtaining the lock
            *guard.lock_var_mut() = true;
            true
        }
    }

    #[inline]
    pub unsafe fn destroy(&self) {}
}

struct ReentrantLock {
    owner: Option<Tcs>,
    count: usize,
}

pub struct ReentrantMutex {
    inner: SpinMutex<WaitVariable<ReentrantLock>>,
}

impl ReentrantMutex {
    pub const fn uninitialized() -> ReentrantMutex {
        ReentrantMutex {
            inner: SpinMutex::new(WaitVariable::new(ReentrantLock { owner: None, count: 0 })),
        }
    }

    #[inline]
    pub unsafe fn init(&self) {}

    #[inline]
    pub unsafe fn lock(&self) {
        let mut guard = self.inner.lock();
        match guard.lock_var().owner {
            Some(tcs) if tcs != thread::current() => {
                // Another thread has the lock, wait
                WaitQueue::wait(guard, || {});
                // Another thread has passed the lock to us
            }
            _ => {
                // We are just now obtaining the lock
                guard.lock_var_mut().owner = Some(thread::current());
                guard.lock_var_mut().count += 1;
            }
        }
    }

    #[inline]
    pub unsafe fn unlock(&self) {
        let mut guard = self.inner.lock();
        if guard.lock_var().count > 1 {
            guard.lock_var_mut().count -= 1;
        } else {
            match WaitQueue::notify_one(guard) {
                Err(mut guard) => {
                    // No other waiters, unlock
                    guard.lock_var_mut().count = 0;
                    guard.lock_var_mut().owner = None;
                }
                Ok(mut guard) => {
                    // There was a thread waiting, just pass the lock
                    if let NotifiedTcs::Single(tcs) = guard.notified_tcs() {
                        guard.lock_var_mut().owner = Some(tcs)
                    } else {
                        unreachable!() // called notify_one
                    }
                }
            }
        }
    }

    #[inline]
    pub unsafe fn try_lock(&self) -> bool {
        let mut guard = try_lock_or_false!(self.inner);
        match guard.lock_var().owner {
            Some(tcs) if tcs != thread::current() => {
                // Another thread has the lock
                false
            }
            _ => {
                // We are just now obtaining the lock
                guard.lock_var_mut().owner = Some(thread::current());
                guard.lock_var_mut().count += 1;
                true
            }
        }
    }

    #[inline]
    pub unsafe fn destroy(&self) {}
}
