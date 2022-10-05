#![deny(unsafe_op_in_unsafe_fn)]
#![allow(dead_code, unused)]

use crate::ffi::CStr;
use crate::io;
use crate::mem;
use crate::sync::Arc;
use crate::num::NonZeroUsize;
use crate::time::Duration;

use super::err2io;

pub const DEFAULT_MIN_STACK_SIZE: usize = 1 * 1024 * 1024;

const TLS_SIZE: usize = 64;
const TLS_ALIGN: usize = 128;

const PTHREAD_SELF_SIZE: usize = 1024;

// Callback when a new thread is spawned
// (making sure the function is never inlined means its use of the stack before
//  its been initialized is minimized)
#[no_mangle]
#[inline(never)]
pub extern "C" fn _start_thread(entry_low: i32, entry_high: i32) {
    let entry_low: u64 = (entry_low as u64) & 0xFFFFFFFF;
    let entry_high: u64 = (entry_high as u64) << 32;
    let entry = entry_low + entry_high;

    let cb = unsafe {
        let cb = entry as *mut ThreadCallback;
        Box::from_raw(cb)
    };

    // Set the thread to a new stack
    unsafe {
        libc::__wasilibc_set_stack_pointer(cb.stack.stack_base as *mut libc::c_void);
    }
    
    // Init the TLS area
    unsafe {
        let mut tls_ptr = cb.stack.tls_ptr;
        if tls_ptr % cb.stack.tls_align != 0 {
            tls_ptr += (cb.stack.tls_align - (tls_ptr % cb.stack.tls_align));
        }
        libc::__wasilibc_init_tls(tls_ptr as *mut libc::c_void);
    }

    // Run the thread
    (cb.callback)();

    // Push the stack back onto the pool
    stack_pool_push(cb.stack);
}

// The reactor will execute any callbacks registered to it when its invoked
// by a calling thread
static REACTOR: SyncLazy<Mutex<Vec<ReactorCallback>>> = SyncLazy::new(|| Default::default());

// Callback when reactor work needs to be processed
// (making sure the function is never inlined means its use of the stack before
//  its been initialized is minimized)
#[no_mangle]
#[inline(never)]
pub extern "C" fn _react(entry_low: i32, entry_high: i32) {
    let entry_low: u64 = (entry_low as u64) & 0xFFFFFFFF;
    let entry_high: u64 = (entry_high as u64) << 32;
    let entry = entry_low + entry_high;

    let cb = unsafe {
        let cb = entry as *mut ReactorCallback;
        mem::ManuallyDrop::new(Box::from_raw(cb))
    };

    // Set the thread to a new stack
    unsafe {
        libc::__wasilibc_set_stack_pointer(cb.stack.stack_base as *mut libc::c_void);
    }

    // Init the TLS area
    unsafe {
        let mut tls_ptr = cb.stack.tls_ptr;
        if tls_ptr % cb.stack.tls_align != 0 {
            tls_ptr += (cb.stack.tls_align - (tls_ptr % cb.stack.tls_align));
        }
        libc::__wasilibc_init_tls(tls_ptr as *mut libc::c_void);
    }

    // Invoke the callback
    (cb.callback)();
}

// Frees memory that was passed to the operating system by
// the program
#[no_mangle]
pub extern "C" fn _free(buf_ptr: u64, _buf_len: u64) {
    unsafe { libc::free(buf_ptr as *mut libc::c_void) };
}

// Allocates memory that will be used to pass data from the
// operating system back to this program
#[no_mangle]
pub extern "C" fn _malloc(len: u64) -> u64 {
    unsafe { libc::malloc(len as usize) as u64 }
}

pub struct Thread
{
    handle: wasi::Tid,
}

struct ThreadStack
{
    stack_start: u64,
    stack_base: u64,
    stack_size: u64,
    tls_ptr: u64,
    tls_size: u64,
    tls_align: u64,
}

impl Drop
for ThreadStack
{
    fn drop(&mut self) {
        _free(self.stack_start, self.stack_size);
        _free(self.tls_ptr, self.tls_size);
    }
}

#[repr(C)]
pub struct ThreadCallback
{
    // Callback that will be invoked
    callback: Box<dyn FnOnce() + 'static>,
    // Memory reserved for the call stack
    stack: ThreadStack,
}

#[repr(C)]
pub struct ReactorCallback
{
    // Callback that will be invoked multiple times
    callback: Box<dyn Fn() + Send + Sync + 'static>,
    // Memory reserved for the call stack
    stack: ThreadStack,
}

use crate::sync::Mutex;
use crate::collections::VecDeque;
use crate::lazy::SyncLazy;

static STACK_POOL: SyncLazy<Mutex<VecDeque<ThreadStack>>> = SyncLazy::new(|| Default::default());

// Takes (or creates) a thread stack from the shared pool
fn stack_pool_pop(stack_size: usize) -> ThreadStack {
    let mut guard = STACK_POOL.lock().unwrap();
    match guard.pop_front() {
        Some(a) => a,
        None => {
            // Create a new empty stack for the the thread
            // (leave some space for the start function, as it may have some variables)
            let stack_start = unsafe {
                libc::malloc(stack_size) as u64
            };
            let stack_base =  stack_start + ((stack_size as u64) - 256);

            let tls_size = unsafe { libc::__wasilibc_tls_size() as usize };
            let tls_align = unsafe { libc::__wasilibc_tls_align() as usize };
            let pthread_self_size = PTHREAD_SELF_SIZE as usize;
            let tls_ptr = unsafe {
                libc::malloc(tls_size + tls_align + pthread_self_size) as u64
            };
            ThreadStack {
                stack_start,
                stack_base,
                stack_size: stack_size as u64,
                tls_ptr,
                tls_size: tls_size as u64,
                tls_align: tls_align as u64,
            }
        }
    }
}

fn stack_pool_push(stack: ThreadStack) {
    let mut guard = STACK_POOL.lock().unwrap();
    guard.push_front(stack);
}

impl Thread {
    pub unsafe fn new(stack_size: usize, p: Box<dyn FnOnce()>) -> io::Result<Thread>
    {
        // Create the callback we will pass back to ourselves in a new thread
        let cb = Box::new(ThreadCallback {
            callback: p,
            stack: stack_pool_pop(stack_size),
        });
        let stack_base = cb.stack.stack_base;
        let stack_start = cb.stack.stack_start;

        // Invoke the thread_spawn callback (if it fails we need to clean up the
        // allocated memory ourselves)
        let handle = unsafe {
            let raw = Box::into_raw(cb) as *mut ThreadCallback;
            wasi::thread_spawn(raw as u64, stack_base as u64, stack_start as u64, wasi::BOOL_FALSE)
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
            callback: Box::new(p),
            stack: stack_pool_pop(DEFAULT_MIN_STACK_SIZE),
        });
        let stack_base = cb.stack.stack_base;
        let stack_start = cb.stack.stack_start;
        let handle = unsafe {
            let raw = Box::into_raw(cb) as *mut ReactorCallback;
            wasi::thread_spawn(raw as u64, stack_base as u64, stack_start as u64, wasi::BOOL_TRUE)
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

    pub unsafe fn current() -> Option<Guard> {
        None
    }
}
