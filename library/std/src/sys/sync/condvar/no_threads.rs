use crate::sys::sync::Mutex;
use crate::thread::sleep;
use crate::time::Duration;

#[cfg(target_has_threads)]
compile_error!("Using no_threads implementation on a target with threads");

pub struct Condvar {}

impl Condvar {
    #[inline]
    pub const fn new() -> Condvar {
        Condvar {}
    }

    #[inline]
    pub fn notify_one(&self) {}

    #[inline]
    pub fn notify_all(&self) {}

    pub unsafe fn wait(&self, _mutex: &Mutex) {
        panic!("condvar wait not supported")
    }

    pub unsafe fn wait_timeout(&self, _mutex: &Mutex, dur: Duration) -> bool {
        sleep(dur);
        false
    }
}
