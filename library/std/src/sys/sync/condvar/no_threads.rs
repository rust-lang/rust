use crate::sys::sync::Mutex;
use crate::time::Duration;

pub struct Condvar {}

impl Condvar {
    #[inline]
    #[cfg_attr(bootstrap, rustc_const_stable(feature = "const_locks", since = "1.63.0"))]
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

    pub unsafe fn wait_timeout(&self, _mutex: &Mutex, _dur: Duration) -> bool {
        panic!("condvar wait not supported");
    }
}
