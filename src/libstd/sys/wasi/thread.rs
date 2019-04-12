use crate::cmp;
use crate::ffi::CStr;
use crate::io;
use crate::sys::cvt;
use crate::sys::{unsupported, Void};
use crate::time::Duration;
use libc;

pub struct Thread(Void);

pub const DEFAULT_MIN_STACK_SIZE: usize = 4096;

impl Thread {
    // unsafe: see thread::Builder::spawn_unchecked for safety requirements
    pub unsafe fn new(_stack: usize, _p: Box<dyn FnOnce()>)
        -> io::Result<Thread>
    {
        unsupported()
    }

    pub fn yield_now() {
        let ret = unsafe { libc::__wasi_sched_yield() };
        debug_assert_eq!(ret, 0);
    }

    pub fn set_name(_name: &CStr) {
        // nope
    }

    pub fn sleep(dur: Duration) {
        let mut secs = dur.as_secs();
        let mut nsecs = dur.subsec_nanos() as i32;

        unsafe {
            while secs > 0 || nsecs > 0 {
                let mut ts = libc::timespec {
                    tv_sec: cmp::min(libc::time_t::max_value() as u64, secs) as libc::time_t,
                    tv_nsec: nsecs,
                };
                secs -= ts.tv_sec as u64;
                cvt(libc::nanosleep(&ts, &mut ts)).unwrap();
                nsecs = 0;
            }
        }
    }

    pub fn join(self) {
        match self.0 {}
    }
}

pub mod guard {
    pub type Guard = !;
    pub unsafe fn current() -> Option<Guard> { None }
    pub unsafe fn init() -> Option<Guard> { None }
}
