#![deny(unsafe_op_in_unsafe_fn)]

use crate::ffi::CStr;
use crate::io;
use crate::mem;
use crate::num::NonZeroUsize;
use crate::sys::unsupported;
use crate::time::Duration;

pub struct Thread(!);

pub const DEFAULT_MIN_STACK_SIZE: usize = 4096;

impl Thread {
    // unsafe: see thread::Builder::spawn_unchecked for safety requirements
    pub unsafe fn new(_stack: usize, _p: Box<dyn FnOnce()>) -> io::Result<Thread> {
        unsupported()
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
            r#type: wasi::EVENTTYPE_CLOCK,
            u: wasi::SubscriptionU { clock },
        };
        unsafe {
            let mut event: wasi::Event = mem::zeroed();
            let res = wasi::poll_oneoff(&in_, &mut event, 1);
            match (res, event) {
                (
                    Ok(1),
                    wasi::Event {
                        userdata: USERDATA, error: 0, r#type: wasi::EVENTTYPE_CLOCK, ..
                    },
                ) => {}
                _ => panic!("thread::sleep(): unexpected result of poll_oneoff"),
            }
        }
    }

    pub fn join(self) {
        self.0
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
