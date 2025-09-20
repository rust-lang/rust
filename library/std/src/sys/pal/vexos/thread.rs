use super::unsupported;
use crate::ffi::CStr;
use crate::io;
use crate::time::{Duration, Instant};

#[expect(dead_code)]
#[path = "../unsupported/thread.rs"]
mod unsupported_thread;
pub use unsupported_thread::{DEFAULT_MIN_STACK_SIZE, available_parallelism};

pub struct Thread(!);

impl Thread {
    // unsafe: see thread::Builder::spawn_unchecked for safety requirements
    pub unsafe fn new(
        _stack: usize,
        _name: Option<&str>,
        _p: Box<dyn FnOnce()>,
    ) -> io::Result<Thread> {
        unsupported()
    }

    pub fn yield_now() {
        unsafe {
            vex_sdk::vexTasksRun();
        }
    }

    pub fn set_name(_name: &CStr) {
        // nope
    }

    pub fn sleep(dur: Duration) {
        let start = Instant::now();

        while start.elapsed() < dur {
            unsafe {
                vex_sdk::vexTasksRun();
            }
        }
    }

    pub fn sleep_until(deadline: Instant) {
        let now = Instant::now();

        if let Some(delay) = deadline.checked_duration_since(now) {
            Self::sleep(delay);
        }
    }

    pub fn join(self) {
        self.0
    }
}

pub(crate) fn current_os_id() -> Option<u64> {
    None
}
