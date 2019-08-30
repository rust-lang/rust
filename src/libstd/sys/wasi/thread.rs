use crate::ffi::CStr;
use crate::io;
use crate::mem;
use crate::sys::{unsupported, Void};
use crate::time::Duration;

use ::wasi::wasi_unstable as wasi;

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
        let ret = wasi::sched_yield();
        debug_assert_eq!(ret, Ok(()));
    }

    pub fn set_name(_name: &CStr) {
        // nope
    }

    pub fn sleep(dur: Duration) {
        let nanos = dur.as_nanos();
        assert!(nanos <= u64::max_value() as u128);

        const CLOCK_ID: wasi::Userdata = 0x0123_45678;

        let clock = wasi::raw::__wasi_subscription_u_clock_t {
            identifier: CLOCK_ID,
            clock_id: wasi::CLOCK_MONOTONIC,
            timeout: nanos as u64,
            precision: 0,
            flags: 0,
        };

        let in_ = [wasi::Subscription {
            userdata: 0,
            type_: wasi::EVENTTYPE_CLOCK,
            u: wasi::raw::__wasi_subscription_u { clock: clock },
        }];
        let (res, event) = unsafe {
            let mut out: [wasi::Event; 1] = mem::zeroed();
            let res = wasi::poll_oneoff(&in_, &mut out);
            (res, out[0])
        };
        match (res, event) {
            (Ok(1), wasi::Event {
                userdata: CLOCK_ID,
                error: 0,
                type_: wasi::EVENTTYPE_CLOCK,
                ..
            }) => {}
            _ => panic!("thread::sleep(): unexpected result of poll_oneoff"),
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
