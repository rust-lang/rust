use crate::sync::atomic::{AtomicUsize, Ordering};
use crate::sys::mutex as mutex_imp;
use crate::sys_common::mutex::MovableMutex;

pub trait CondvarCheck {
    type Check;
}

/// For boxed mutexes, a `Condvar` will check it's only ever used with the same
/// mutex, based on its (stable) address.
impl CondvarCheck for Box<mutex_imp::Mutex> {
    type Check = SameMutexCheck;
}

pub struct SameMutexCheck {
    addr: AtomicUsize,
}

#[allow(dead_code)]
impl SameMutexCheck {
    pub const fn new() -> Self {
        Self { addr: AtomicUsize::new(0) }
    }
    pub fn verify(&self, mutex: &MovableMutex) {
        let addr = mutex.raw() as *const mutex_imp::Mutex as usize;
        match self.addr.compare_exchange(0, addr, Ordering::SeqCst, Ordering::SeqCst) {
            Ok(_) => {}               // Stored the address
            Err(n) if n == addr => {} // Lost a race to store the same address
            _ => panic!("attempted to use a condition variable with two mutexes"),
        }
    }
}

/// Unboxed mutexes may move, so `Condvar` can not require its address to stay
/// constant.
impl CondvarCheck for mutex_imp::Mutex {
    type Check = NoCheck;
}

pub struct NoCheck;

#[allow(dead_code)]
impl NoCheck {
    pub const fn new() -> Self {
        Self
    }
    pub fn verify(&self, _: &MovableMutex) {}
}
