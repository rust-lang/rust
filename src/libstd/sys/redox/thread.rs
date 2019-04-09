use crate::ffi::CStr;
use crate::io;
use crate::mem;
use crate::sys_common::thread::start_thread;
use crate::sys::{cvt, syscall};
use crate::time::Duration;

pub const DEFAULT_MIN_STACK_SIZE: usize = 2 * 1024 * 1024;

pub struct Thread {
    id: usize,
}

// Some platforms may have pthread_t as a pointer in which case we still want
// a thread to be Send/Sync
unsafe impl Send for Thread {}
unsafe impl Sync for Thread {}

impl Thread {
    // unsafe: see thread::Builder::spawn_unchecked for safety requirements
    pub unsafe fn new(_stack: usize, p: Box<dyn FnOnce()>) -> io::Result<Thread> {
        let p = box p;

        let id = cvt(syscall::clone(syscall::CLONE_VM | syscall::CLONE_FS | syscall::CLONE_FILES))?;
        if id == 0 {
            start_thread(&*p as *const _ as *mut _);
            let _ = syscall::exit(0);
            panic!("thread failed to exit");
        } else {
            mem::forget(p);
            Ok(Thread { id })
        }
    }

    pub fn yield_now() {
        let ret = syscall::sched_yield().expect("failed to sched_yield");
        debug_assert_eq!(ret, 0);
    }

    pub fn set_name(_name: &CStr) {

    }

    pub fn sleep(dur: Duration) {
        let mut secs = dur.as_secs();
        let mut nsecs = dur.subsec_nanos() as i32;

        // If we're awoken with a signal then the return value will be -1 and
        // nanosleep will fill in `ts` with the remaining time.
        while secs > 0 || nsecs > 0 {
            let req = syscall::TimeSpec {
                tv_sec: secs as i64,
                tv_nsec: nsecs,
            };
            secs -= req.tv_sec as u64;
            let mut rem = syscall::TimeSpec::default();
            if syscall::nanosleep(&req, &mut rem).is_err() {
                secs += rem.tv_sec as u64;
                nsecs = rem.tv_nsec;
            } else {
                nsecs = 0;
            }
        }
    }

    pub fn join(self) {
        let mut status = 0;
        syscall::waitpid(self.id, &mut status, 0).unwrap();
    }

    pub fn id(&self) -> usize { self.id }

    pub fn into_id(self) -> usize {
        let id = self.id;
        mem::forget(self);
        id
    }
}

pub mod guard {
    pub type Guard = !;
    pub unsafe fn current() -> Option<Guard> { None }
    pub unsafe fn init() -> Option<Guard> { None }
}
