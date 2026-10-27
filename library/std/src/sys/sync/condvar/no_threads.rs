use crate::sys::sync::Mutex;
use crate::thread::sleep;
use crate::time::Duration;

#[cfg(all(
    target_has_threads,
    // VEXos is only "target_has_threads" because it allows user interrupt handlers which preempt
    // the main thread, but they are forbidden from accessing sync primitives (the same as UNIX
    // signal handlers).
    not(target_os = "vexos"),
))]
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
