use crate::sys::mutex::Mutex;
use crate::time::Duration;

pub struct Condvar { }

impl Condvar {
    pub const fn new() -> Condvar {
        Condvar { }
    }

    #[inline]
    pub unsafe fn init(&mut self) {}

    #[inline]
    pub unsafe fn notify_one(&self) {
    }

    #[inline]
    pub unsafe fn notify_all(&self) {
    }

    pub unsafe fn wait(&self, _mutex: &Mutex) {
        panic!("can't block with web assembly")
    }

    pub unsafe fn wait_timeout(&self, _mutex: &Mutex, _dur: Duration) -> bool {
        panic!("can't block with web assembly");
    }

    #[inline]
    pub unsafe fn destroy(&self) {
    }
}
