use crate::ptr;
use crate::sync::atomic::{AtomicPtr, Ordering};
use crate::sys::locks as imp;
use crate::sys_common::lazy_box::{LazyBox, LazyInit};
use crate::sys_common::mutex::MovableMutex;

pub trait CondvarCheck {
    type Check;
}

/// For boxed mutexes, a `Condvar` will check it's only ever used with the same
/// mutex, based on its (stable) address.
impl<T: LazyInit> CondvarCheck for LazyBox<T> {
    type Check = SameMutexCheck;
}

pub struct SameMutexCheck {
    addr: AtomicPtr<()>,
}

#[allow(dead_code)]
impl SameMutexCheck {
    pub const fn new() -> Self {
        Self { addr: AtomicPtr::new(ptr::null_mut()) }
    }
    pub fn verify(&self, mutex: &MovableMutex) {
        let addr = mutex.raw() as *const imp::Mutex as *const () as *mut _;
        // Relaxed is okay here because we never read through `self.addr`, and only use it to
        // compare addresses.
        match self.addr.compare_exchange(
            ptr::null_mut(),
            addr,
            Ordering::Relaxed,
            Ordering::Relaxed,
        ) {
            Ok(_) => {}               // Stored the address
            Err(n) if n == addr => {} // Lost a race to store the same address
            _ => panic!("attempted to use a condition variable with two mutexes"),
        }
    }
}

/// Unboxed mutexes may move, so `Condvar` can not require its address to stay
/// constant.
impl CondvarCheck for imp::Mutex {
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
