use crate::sync::atomic::{AtomicUsize, Ordering};
use crate::sys::mutex as mutex_imp;
use crate::sys_common::mutex::MovableMutex;

/// A `Condvar` will check it's only ever used with the same mutex, based on
/// its (stable) address.
pub struct CondvarCheck {
    addr: AtomicUsize,
}

impl CondvarCheck {
    pub const fn new() -> Self {
        Self { addr: AtomicUsize::new(0) }
    }
    pub fn verify(&self, mutex: &MovableMutex) {
        let addr = mutex.raw() as *const mutex_imp::Mutex as usize;
        match self.addr.compare_and_swap(0, addr, Ordering::SeqCst) {
            0 => {}              // Stored the address
            n if n == addr => {} // Lost a race to store the same address
            _ => panic!("attempted to use a condition variable with two mutexes"),
        }
    }
}
