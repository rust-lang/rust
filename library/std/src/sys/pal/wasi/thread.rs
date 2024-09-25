#![forbid(unsafe_op_in_unsafe_fn)]

use crate::ffi::CStr;
use crate::num::NonZero;
use crate::sys::unsupported;
use crate::time::Duration;
use crate::{io, mem};

cfg_if::cfg_if! {
    if #[cfg(target_feature = "atomics")] {
        use crate::cmp;
        use crate::ptr;
        use crate::sys::os;
        // Add a few symbols not in upstream `libc` just yet.
        mod libc {
            pub use crate::ffi;
            pub use crate::mem;
            pub use libc::*;

            // defined in wasi-libc
            // https://github.com/WebAssembly/wasi-libc/blob/a6f871343313220b76009827ed0153586361c0d5/libc-top-half/musl/include/alltypes.h.in#L108
            #[repr(C)]
            union pthread_attr_union {
                __i: [ffi::c_int; if mem::size_of::<ffi::c_long>() == 8 { 14 } else { 9 }],
                __vi: [ffi::c_int; if mem::size_of::<ffi::c_long>() == 8 { 14 } else { 9 }],
                __s: [ffi::c_ulong; if mem::size_of::<ffi::c_long>() == 8 { 7 } else { 9 }],
            }

            #[repr(C)]
            pub struct pthread_attr_t {
                __u: pthread_attr_union,
            }

            #[allow(non_camel_case_types)]
            pub type pthread_t = *mut ffi::c_void;

            extern "C" {
                pub fn pthread_create(
                    native: *mut pthread_t,
                    attr: *const pthread_attr_t,
                    f: extern "C" fn(*mut ffi::c_void) -> *mut ffi::c_void,
                    value: *mut ffi::c_void,
                ) -> ffi::c_int;
                pub fn pthread_join(native: pthread_t, value: *mut *mut ffi::c_void) -> ffi::c_int;
                pub fn pthread_attr_init(attrp: *mut pthread_attr_t) -> ffi::c_int;
                pub fn pthread_attr_setstacksize(
                    attr: *mut pthread_attr_t,
                    stack_size: libc::size_t,
                ) -> ffi::c_int;
                pub fn pthread_attr_destroy(attr: *mut pthread_attr_t) -> ffi::c_int;
                pub fn pthread_detach(thread: pthread_t) -> ffi::c_int;
            }
        }

        pub struct Thread {
            id: libc::pthread_t,
        }

        impl Drop for Thread {
            fn drop(&mut self) {
                let ret = unsafe { libc::pthread_detach(self.id) };
                debug_assert_eq!(ret, 0);
            }
        }
    } else {
        pub struct Thread(!);
    }
}

pub const DEFAULT_MIN_STACK_SIZE: usize = 64 * 1024;

impl Thread {
    // unsafe: see thread::Builder::spawn_unchecked for safety requirements
    cfg_if::cfg_if! {
        if #[cfg(target_feature = "atomics")] {
            pub unsafe fn new(stack: usize, p: Box<dyn FnOnce()>) -> io::Result<Thread> {
                let p = Box::into_raw(Box::new(p));
                let mut native: libc::pthread_t = unsafe { mem::zeroed() };
                let mut attr: libc::pthread_attr_t = unsafe { mem::zeroed() };
                assert_eq!(unsafe { libc::pthread_attr_init(&mut attr) }, 0);

                let stack_size = cmp::max(stack, DEFAULT_MIN_STACK_SIZE);

                match unsafe { libc::pthread_attr_setstacksize(&mut attr, stack_size) } {
                    0 => {}
                    n => {
                        assert_eq!(n, libc::EINVAL);
                        // EINVAL means |stack_size| is either too small or not a
                        // multiple of the system page size. Because it's definitely
                        // >= PTHREAD_STACK_MIN, it must be an alignment issue.
                        // Round up to the nearest page and try again.
                        let page_size = os::page_size();
                        let stack_size =
                            (stack_size + page_size - 1) & (-(page_size as isize - 1) as usize - 1);
                        assert_eq!(unsafe { libc::pthread_attr_setstacksize(&mut attr, stack_size) }, 0);
                    }
                };

                let ret = unsafe { libc::pthread_create(&mut native, &attr, thread_start, p as *mut _) };
                // Note: if the thread creation fails and this assert fails, then p will
                // be leaked. However, an alternative design could cause double-free
                // which is clearly worse.
                assert_eq!(unsafe {libc::pthread_attr_destroy(&mut attr) }, 0);

                return if ret != 0 {
                    // The thread failed to start and as a result p was not consumed. Therefore, it is
                    // safe to reconstruct the box so that it gets deallocated.
                    unsafe { drop(Box::from_raw(p)); }
                    Err(io::Error::from_raw_os_error(ret))
                } else {
                    Ok(Thread { id: native })
                };

                extern "C" fn thread_start(main: *mut libc::c_void) -> *mut libc::c_void {
                    unsafe {
                        // Finally, let's run some code.
                        Box::from_raw(main as *mut Box<dyn FnOnce()>)();
                    }
                    ptr::null_mut()
                }
            }
        } else {
            pub unsafe fn new(_stack: usize, _p: Box<dyn FnOnce()>) -> io::Result<Thread> {
                unsupported()
            }
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
        let mut nanos = dur.as_nanos();
        while nanos > 0 {
            const USERDATA: wasi::Userdata = 0x0123_45678;

            let clock = wasi::SubscriptionClock {
                id: wasi::CLOCKID_MONOTONIC,
                timeout: u64::try_from(nanos).unwrap_or(u64::MAX),
                precision: 0,
                flags: 0,
            };
            nanos -= u128::from(clock.timeout);

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
    }

    pub fn join(self) {
        cfg_if::cfg_if! {
            if #[cfg(target_feature = "atomics")] {
                let id = mem::ManuallyDrop::new(self).id;
                let ret = unsafe { libc::pthread_join(id, ptr::null_mut()) };
                if ret != 0 {
                    rtabort!("failed to join thread: {}", io::Error::from_raw_os_error(ret));
                }
            } else {
                self.0
            }
        }
    }
}

pub fn available_parallelism() -> io::Result<NonZero<usize>> {
    unsupported()
}
