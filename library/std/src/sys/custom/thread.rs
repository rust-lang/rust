#![unstable(issue = "none", feature = "std_internals")]
#![allow(missing_docs)]

use crate::custom_os_impl;
use crate::ffi::CStr;
use crate::io;
use crate::num::NonZeroUsize;
use crate::time::Duration;

/// Inner content of [`crate::thread::Thread`]
#[derive(Debug)]
pub struct Thread(pub Box<dyn ThreadApi>);

pub const DEFAULT_MIN_STACK_SIZE: usize = 4096;

/// Object-oriented manipulation of a [`Thread`]
pub trait ThreadApi: crate::fmt::Debug {
    // self will be dropped upon return
    fn join(&self);
}

impl Thread {
    // unsafe: see thread::Builder::spawn_unchecked for safety requirements
    pub unsafe fn new(stack: usize, p: Box<dyn FnOnce()>) -> io::Result<Thread> {
        custom_os_impl!(thread, new, stack, p)
    }

    pub fn yield_now() {
        custom_os_impl!(thread, yield_now)
    }

    pub fn set_name(name: &CStr) {
        custom_os_impl!(thread, set_name, name)
    }

    pub fn sleep(dur: Duration) {
        custom_os_impl!(thread, sleep, dur)
    }

    pub fn join(self) {
        self.0.join()
    }
}

pub fn available_parallelism() -> io::Result<NonZeroUsize> {
    custom_os_impl!(thread, available_parallelism)
}

pub mod guard {
    pub type Guard = !;
    pub unsafe fn current() -> Option<Guard> {
        None
    }
    pub unsafe fn init() -> Option<Guard> {
        None
    }
}
