#![deny(unsafe_op_in_unsafe_fn)]

use crate::ffi::CStr;
use crate::io;
use crate::mem;
use crate::sync::Arc;
use crate::num::NonZeroUsize;
use crate::time::Duration;

use super::err2io;

pub const DEFAULT_MIN_STACK_SIZE: usize = 4096;

// Callback when a new thread is spawned
#[no_mangle]
pub extern "C" fn _thread_start(entry: u64) {
    let cb = unsafe {
        let cb = entry as *mut ThreadCallback;
        Box::from_raw(cb)
    };

    (cb.callback)();
}

// Callback when reactor work is initiated
// (this leaks the memory until _reactor_finish is called)
#[no_mangle]
pub extern "C" fn _reactor_work(entry: u64) {
    let cb = unsafe {
        let cb = entry as *mut ReactorCallback;
        mem::ManuallyDrop::new(Box::from_raw(cb))
    };
    (cb.callback)();
}

// Cleans up all the resources associated with this reactor
#[no_mangle]
pub extern "C" fn _reactor_finish(entry: u64) {
    let cb = unsafe {
        let cb = entry as *mut ReactorCallback;
        Box::from_raw(cb)
    };
    mem::drop(cb);
}

// Frees memory that was passed to the operating system by
// the program
#[no_mangle]
pub extern "C" fn _free(buf_ptr: u64, buf_len: u64) {
    unsafe {
        let data = Vec::from_raw_parts(buf_ptr as *mut u8, buf_len as usize, buf_len as usize);
        mem::drop(data);
    }
}

// Allocates memory that will be used to pass data from the
// operating system back to this program
#[no_mangle]
pub extern "C" fn _malloc(len: u64) -> u64 {
    let mut buf = Vec::with_capacity(len as usize);
    let ptr: *mut u8 = buf.as_mut_ptr();
    mem::forget(buf);
    return ptr as u64;
}

pub struct Thread
{
    handle: wasi::Tid,
}

#[repr(C)]
pub struct ThreadCallback
{
    callback: Box<dyn FnOnce() + 'static>
}

#[repr(C)]
#[derive(Clone)]
pub struct ReactorCallback
{
    callback: Arc<dyn Fn() + Send + Sync + 'static>
}

impl Thread {
    pub unsafe fn new(_stack: usize, p: Box<dyn FnOnce()>) -> io::Result<Thread>
    {
        let cb = Box::new(ThreadCallback {
            callback: p
        });
        let handle = unsafe {
            let raw = Box::into_raw(cb) as *mut ThreadCallback;
            wasi::thread_spawn("_thread_start", "_malloc", raw as u64, wasi::BOOL_FALSE)
                .map_err(err2io)?
        };

        Ok(
            Thread {
                handle
            }
        )
    }

    pub unsafe fn new_reactor<F>(p: F) -> io::Result<Thread>
    where F: Fn() + Send + Sync + 'static
    {
        let cb = Box::new(ReactorCallback {
            callback: Arc::new(p)
        });
        let handle = unsafe {
            let raw = Box::into_raw(cb) as *mut ReactorCallback;
            wasi::thread_spawn("_reactor_work", "_malloc", raw as u64, wasi::BOOL_TRUE)
                .map_err(err2io)?
        };

        Ok(
            Thread {
                handle
            }
        )
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

pub fn available_parallelism() -> io::Result<NonZeroUsize> {
    let val: NonZeroUsize = unsafe {
        wasi::thread_parallelism().map_err(err2io)?.try_into().unwrap()
    };
    Ok(val)
}

pub mod guard {
    pub type Guard = !;
    pub unsafe fn init() -> Option<Guard> {
        None
    }
}
    