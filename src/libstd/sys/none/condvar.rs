use sys::mutex::Mutex;
use time::Duration;

pub struct Condvar(());

unsafe impl Send for Condvar {}
unsafe impl Sync for Condvar {}

impl Condvar {
    #[inline] pub const fn new() -> Condvar { Condvar(()) }
    #[inline] pub unsafe fn init(&mut self) {}
    #[inline] pub unsafe fn notify_one(&self) {}
    #[inline] pub unsafe fn notify_all(&self) {}
    #[inline] pub unsafe fn wait(&self, _mutex: &Mutex) {}
    #[inline] pub unsafe fn wait_timeout(&self, _mutex: &Mutex, _dur: Duration) -> bool { true }
    #[inline] pub unsafe fn destroy(&self) {}
}
