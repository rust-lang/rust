#![allow(dead_code)]

use super::hermit_abi;
use crate::ffi::CStr;
use crate::mem::ManuallyDrop;
use crate::num::NonZero;
use crate::time::Duration;
use crate::{io, ptr};

pub type Tid = hermit_abi::Tid;

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
        let p = Box::into_raw(Box::new(p));
        let tid = unsafe {
            hermit_abi::spawn2(
                thread_start,
                p.expose_provenance(),
                hermit_abi::Priority::into(hermit_abi::NORMAL_PRIO),
                stack,
                core_id,
            )
        };

        return if tid == 0 {
            // The thread failed to start and as a result p was not consumed. Therefore, it is
            // safe to reconstruct the box so that it gets deallocated.
            unsafe {
                drop(Box::from_raw(p));
            }
            Err(io::const_error!(io::ErrorKind::Uncategorized, "Unable to create thread!"))
        } else {
            Ok(Thread { tid })
        };

        extern "C" fn thread_start(main: usize) {
            unsafe {
                // Finally, let's run some code.
                Box::from_raw(ptr::with_exposed_provenance::<Box<dyn FnOnce()>>(main).cast_mut())();

                // run all destructors
                crate::sys::thread_local::destructors::run();
                crate::rt::thread_cleanup();
            }
        }
    }

    pub unsafe fn new(stack: usize, p: Box<dyn FnOnce()>) -> io::Result<Thread> {
        unsafe {
            Thread::new_with_coreid(stack, p, -1 /* = no specific core */)
        }
    }

    #[inline]
    pub fn yield_now() {
        unsafe {
            hermit_abi::yield_now();
        }
    }

    #[inline]
    pub fn set_name(_name: &CStr) {
        // nope
    }

    #[inline]
    pub fn sleep(dur: Duration) {
        let micros = dur.as_micros() + if dur.subsec_nanos() % 1_000 > 0 { 1 } else { 0 };
        let micros = u64::try_from(micros).unwrap_or(u64::MAX);

        unsafe {
            hermit_abi::usleep(micros);
        }
    }

    pub fn join(self) {
        unsafe {
            let _ = hermit_abi::join(self.tid);
        }
    }

    #[inline]
    pub fn id(&self) -> Tid {
        self.tid
    }

    #[inline]
    pub fn into_id(self) -> Tid {
        ManuallyDrop::new(self).tid
    }
}

pub fn available_parallelism() -> io::Result<NonZero<usize>> {
    unsafe { Ok(NonZero::new_unchecked(hermit_abi::available_parallelism())) }
}
