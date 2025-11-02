use crate::ffi::CStr;
use crate::io;
use crate::num::NonZero;
use crate::time::Duration;

pub struct Thread(!);

pub const DEFAULT_MIN_STACK_SIZE: usize = 64 * 1024;

impl Thread {
    // unsafe: see thread::Builder::spawn_unchecked for safety requirements
    pub unsafe fn new(
        _stack: usize,
        _name: Option<&str>,
        _p: Box<dyn FnOnce()>,
    ) -> io::Result<Thread> {
        Err(io::Error::UNSUPPORTED_PLATFORM)
    }

    pub fn join(self) {
        self.0
    }
}

pub fn available_parallelism() -> io::Result<NonZero<usize>> {
    Err(io::Error::UNKNOWN_THREAD_COUNT)
}

pub fn current_os_id() -> Option<u64> {
    None
}

pub fn yield_now() {
    // do nothing
}

pub fn set_name(_name: &CStr) {
    // nope
}

pub fn sleep(_dur: Duration) {
    panic!("can't sleep");
}
