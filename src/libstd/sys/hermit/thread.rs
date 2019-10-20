#![allow(dead_code)]

use crate::ffi::CStr;
use crate::io;
use crate::sys::hermit::abi;
use crate::time::Duration;
use crate::mem;
use crate::fmt;
use core::u32;

use crate::sys_common::thread::*;

pub type Tid = abi::Tid;

/// Priority of a task
#[derive(PartialEq, Eq, PartialOrd, Ord, Debug, Clone, Copy)]
pub struct Priority(u8);

impl Priority {
    pub const fn into(self) -> u8 {
        self.0
    }

    pub const fn from(x: u8) -> Self {
        Priority(x)
    }
}

impl fmt::Display for Priority {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

pub const NORMAL_PRIO: Priority = Priority::from(2);

pub struct Thread {
    tid: Tid
}

unsafe impl Send for Thread {}
unsafe impl Sync for Thread {}

pub const DEFAULT_MIN_STACK_SIZE: usize = 262144;

impl Thread {
    pub unsafe fn new_with_coreid(_stack: usize, p: Box<dyn FnOnce()>, core_id: isize)
        -> io::Result<Thread>
    {
        let p = box p;
        let mut tid: Tid = u32::MAX;
        let ret = abi::spawn(&mut tid as *mut Tid, thread_start,
                            &*p as *const _ as *const u8 as usize,
                            Priority::into(NORMAL_PRIO), core_id);

        return if ret == 0 {
            mem::forget(p); // ownership passed to pthread_create
            Ok(Thread { tid: tid })
        } else {
            Err(io::Error::new(io::ErrorKind::Other, "Unable to create thread!"))
        };

        extern fn thread_start(main: usize) {
            unsafe {
                start_thread(main as *mut u8);
            }
        }
    }

    pub unsafe fn new(stack: usize, p: Box<dyn FnOnce()>)
        -> io::Result<Thread>
    {
        Thread::new_with_coreid(stack, p, -1 /* = no specific core */)
    }

    #[inline]
    pub fn yield_now() {
        unsafe {
            abi::yield_now();
        }
    }

    #[inline]
    pub fn set_name(_name: &CStr) {
        // nope
    }

    #[inline]
    pub fn sleep(dur: Duration) {
        unsafe {
            abi::usleep(dur.as_micros() as u64);
        }
    }

    pub fn join(self) {
        unsafe {
            let _ = abi::join(self.tid);
        }
    }

    #[inline]
    pub fn id(&self) -> Tid { self.tid }

    #[inline]
    pub fn into_id(self) -> Tid {
        let id = self.tid;
        mem::forget(self);
        id
    }
}

pub mod guard {
    pub type Guard = !;
    pub unsafe fn current() -> Option<Guard> { None }
    pub unsafe fn init() -> Option<Guard> { None }
}
