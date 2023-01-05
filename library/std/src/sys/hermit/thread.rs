#![allow(dead_code)]

use super::unsupported;
use crate::ffi::CStr;
use crate::io;
use crate::mem;
use crate::num::NonZeroUsize;
use crate::ptr;
use crate::sys::hermit::abi;
use crate::sys::hermit::thread_local_dtor::run_dtors;
use crate::time::Duration;

pub type Tid = abi::Tid;

pub struct Thread {
    tid: Tid,
}

unsafe impl Send for Thread {}
unsafe impl Sync for Thread {}

pub const DEFAULT_MIN_STACK_SIZE: usize = 1 << 20;

impl Thread {
    pub unsafe fn new_with_coreid(
        stack: usize,
        p: Box<dyn FnOnce()>,
        core_id: isize,
    ) -> io::Result<Thread> {
        let p = Box::into_raw(box p);
        let tid = abi::spawn2(
            thread_start,
            p as usize,
            abi::Priority::into(abi::NORMAL_PRIO),
            stack,
            core_id,
        );

        return if tid == 0 {
            // The thread failed to start and as a result p was not consumed. Therefore, it is
            // safe to reconstruct the box so that it gets deallocated.
            drop(Box::from_raw(p));
            Err(io::const_io_error!(io::ErrorKind::Uncategorized, "Unable to create thread!"))
        } else {
            Ok(Thread { tid: tid })
        };

        extern "C" fn thread_start(main: usize) {
            unsafe {
                // Finally, let's run some code.
                Box::from_raw(ptr::from_exposed_addr::<Box<dyn FnOnce()>>(main).cast_mut())();

                // run all destructors
                run_dtors();
            }
        }
    }

    pub unsafe fn new(stack: usize, p: Box<dyn FnOnce()>) -> io::Result<Thread> {
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
    pub fn id(&self) -> Tid {
        self.tid
    }

    #[inline]
    pub fn into_id(self) -> Tid {
        let id = self.tid;
        mem::forget(self);
        id
    }
}

pub fn available_parallelism() -> io::Result<NonZeroUsize> {
    unsupported()
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
