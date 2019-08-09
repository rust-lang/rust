use crate::cmp;
use crate::ffi::CStr;
use crate::io;
use crate::mem;
use crate::ptr;
use crate::sys::cloudabi::abi;
use crate::sys::time::checked_dur2intervals;
use crate::sys_common::thread::*;
use crate::time::Duration;

pub const DEFAULT_MIN_STACK_SIZE: usize = 2 * 1024 * 1024;

pub struct Thread {
    id: libc::pthread_t,
}

// CloudABI has pthread_t as a pointer in which case we still want
// a thread to be Send/Sync
unsafe impl Send for Thread {}
unsafe impl Sync for Thread {}

impl Thread {
    // unsafe: see thread::Builder::spawn_unchecked for safety requirements
    pub unsafe fn new(stack: usize, p: Box<dyn FnOnce()>) -> io::Result<Thread> {
        let p = box p;
        let mut native: libc::pthread_t = mem::zeroed();
        let mut attr: libc::pthread_attr_t = mem::zeroed();
        assert_eq!(libc::pthread_attr_init(&mut attr), 0);

        let stack_size = cmp::max(stack, min_stack_size(&attr));
        assert_eq!(libc::pthread_attr_setstacksize(&mut attr, stack_size), 0);

        let ret = libc::pthread_create(&mut native, &attr, thread_start, &*p as *const _ as *mut _);
        assert_eq!(libc::pthread_attr_destroy(&mut attr), 0);

        return if ret != 0 {
            Err(io::Error::from_raw_os_error(ret))
        } else {
            mem::forget(p); // ownership passed to pthread_create
            Ok(Thread { id: native })
        };

        extern "C" fn thread_start(main: *mut libc::c_void) -> *mut libc::c_void {
            unsafe {
                start_thread(main as *mut u8);
            }
            ptr::null_mut()
        }
    }

    pub fn yield_now() {
        let ret = unsafe { abi::thread_yield() };
        debug_assert_eq!(ret, abi::errno::SUCCESS);
    }

    pub fn set_name(_name: &CStr) {
        // CloudABI has no way to set a thread name.
    }

    pub fn sleep(dur: Duration) {
        let timeout = checked_dur2intervals(&dur)
            .expect("overflow converting duration to nanoseconds");
        unsafe {
            let subscription = abi::subscription {
                type_: abi::eventtype::CLOCK,
                union: abi::subscription_union {
                    clock: abi::subscription_clock {
                        clock_id: abi::clockid::MONOTONIC,
                        timeout,
                        ..mem::zeroed()
                    },
                },
                ..mem::zeroed()
            };
            let mut event: abi::event = mem::uninitialized();
            let mut nevents: usize = mem::uninitialized();
            let ret = abi::poll(&subscription, &mut event, 1, &mut nevents);
            assert_eq!(ret, abi::errno::SUCCESS);
            assert_eq!(event.error, abi::errno::SUCCESS);
        }
    }

    pub fn join(self) {
        unsafe {
            let ret = libc::pthread_join(self.id, ptr::null_mut());
            mem::forget(self);
            assert!(
                ret == 0,
                "failed to join thread: {}",
                io::Error::from_raw_os_error(ret)
            );
        }
    }
}

impl Drop for Thread {
    fn drop(&mut self) {
        let ret = unsafe { libc::pthread_detach(self.id) };
        debug_assert_eq!(ret, 0);
    }
}

#[cfg_attr(test, allow(dead_code))]
pub mod guard {
    pub type Guard = !;
    pub unsafe fn current() -> Option<Guard> {
        None
    }
    pub unsafe fn init() -> Option<Guard> {
        None
    }
}

fn min_stack_size(_: *const libc::pthread_attr_t) -> usize {
    libc::PTHREAD_STACK_MIN
}
