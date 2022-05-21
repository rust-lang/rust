#![deny(unsafe_op_in_unsafe_fn)]

use crate::ffi::CStr;
use crate::io;
use crate::mem;
use crate::num::NonZeroUsize;
use crate::sys::unsupported;
use crate::time::Duration;

use super::err2io;

pub const DEFAULT_MIN_STACK_SIZE: usize = 4096;

pub struct Thread
{
    handle: wasi::Tid,
}

impl Thread {
    #[cfg(not(target_feature = "atomics"))]
    pub unsafe fn new(_stack: usize, _p: Box<dyn FnOnce()>) -> io::Result<Thread> {
        unsupported()
    }

    #[cfg(target_feature = "atomics")]
    pub unsafe fn new(_stack: usize, p: Box<dyn FnOnce()>) -> io::Result<Thread> {
        unsafe {
            let r = p as *mut _;
            std::mem::forget(p);
            let handle = wasi::thread_spawn("thread_start", r as u64, wasi::Bool::False)
                .map_err(err2io)?;
            Ok(
                Thread {
                    handle
                }
            )
        }
    }

    #[cfg(target_feature = "atomics")]
    extern "C" fn thread_start(entry: u64) {
        unsafe {
            let p = entry as *mut Box<dyn FnOnce()>;
            let p = Box::from_raw(p);
            p();
        }
    }

    pub fn yield_now() {
        let ret = unsafe { wasi::sched_yield() };
        debug_assert_eq!(ret, Ok(()));
    }

    pub fn set_name(_name: &CStr) {
        // nope
    }

    pub fn sleep(dur: Duration) {
        let nanos = dur.as_nanos();
        assert!(nanos <= u64::MAX as u128);

        const USERDATA: wasi::Userdata = 0x0123_45678;

        let clock = wasi::SubscriptionClock {
            id: wasi::CLOCKID_MONOTONIC,
            timeout: nanos as u64,
            precision: 0,
            flags: 0,
        };

        let in_ = wasi::Subscription {
            userdata: USERDATA,
            u: wasi::SubscriptionU { tag: 0, u: wasi::SubscriptionUU { clock } },
        };
        unsafe {
            let mut event: wasi::Event = mem::zeroed();
            let res = wasi::poll_oneoff(&in_, &mut event, 1);
            match (res, event) {
                (
                    Ok(1),
                    wasi::Event {
                        userdata: USERDATA,
                        error: wasi::ERRNO_SUCCESS,
                        type_: wasi::EVENTTYPE_CLOCK,
                        ..
                    },
                ) => {}
                _ => panic!("thread::sleep(): unexpected result of poll_oneoff"),
            }
        }
    }

    pub fn join(self) {
        unsafe {
            let ret = wasi::thread_join(self.handle).map_err(err2io);
            mem::forget(self);
            assert!(ret.is_ok(), "failed to join thread: {}", ret.unwrap_err());
        }
    }

    #[allow(dead_code)]
    pub fn id(&self) -> u32 {
        self.handle
    }

    #[allow(dead_code)]
    pub fn into_id(self) -> u32 {
        let id = self.handle;
        mem::forget(self);
        id
    }
}

#[cfg(target_feature = "atomics")]
pub fn available_parallelism() -> io::Result<NonZeroUsize> {
    unsafe {
        Ok (
            wasi::thread_parallelism().map_err(err2io)?
        )
    }
}

#[cfg(not(target_feature = "atomics"))]
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
    