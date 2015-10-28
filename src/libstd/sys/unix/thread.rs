// Copyright 2014-2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![allow(dead_code)]

use prelude::v1::*;

use alloc::boxed::FnBox;
use cmp;
use ffi::CString;
use io;
use libc::consts::os::posix01::PTHREAD_STACK_MIN;
use libc;
use mem;
use ptr;
use sys::os;
use time::Duration;

use sys_common::thread::*;

pub struct Thread {
    id: libc::pthread_t,
}

// Some platforms may have pthread_t as a pointer in which case we still want
// a thread to be Send/Sync
unsafe impl Send for Thread {}
unsafe impl Sync for Thread {}

impl Thread {
    pub unsafe fn new<'a>(stack: usize, p: Box<FnBox() + 'a>)
                          -> io::Result<Thread> {
        let p = box p;
        let mut native: libc::pthread_t = mem::zeroed();
        let mut attr: libc::pthread_attr_t = mem::zeroed();
        assert_eq!(pthread_attr_init(&mut attr), 0);

        let stack_size = cmp::max(stack, min_stack_size(&attr));
        match pthread_attr_setstacksize(&mut attr, stack_size as libc::size_t) {
            0 => {}
            n => {
                assert_eq!(n, libc::EINVAL);
                // EINVAL means |stack_size| is either too small or not a
                // multiple of the system page size.  Because it's definitely
                // >= PTHREAD_STACK_MIN, it must be an alignment issue.
                // Round up to the nearest page and try again.
                let page_size = os::page_size();
                let stack_size = (stack_size + page_size - 1) &
                                 (-(page_size as isize - 1) as usize - 1);
                let stack_size = stack_size as libc::size_t;
                assert_eq!(pthread_attr_setstacksize(&mut attr, stack_size), 0);
            }
        };

        let ret = pthread_create(&mut native, &attr, thread_start,
                                 &*p as *const _ as *mut _);
        assert_eq!(pthread_attr_destroy(&mut attr), 0);

        return if ret != 0 {
            Err(io::Error::from_raw_os_error(ret))
        } else {
            mem::forget(p); // ownership passed to pthread_create
            Ok(Thread { id: native })
        };

        extern fn thread_start(main: *mut libc::c_void) -> *mut libc::c_void {
            unsafe { start_thread(main); }
            ptr::null_mut()
        }
    }

    pub fn yield_now() {
        let ret = unsafe { sched_yield() };
        debug_assert_eq!(ret, 0);
    }

    #[cfg(any(target_os = "linux", target_os = "android"))]
    pub fn set_name(name: &str) {
        // pthread wrapper only appeared in glibc 2.12, so we use syscall
        // directly.
        extern {
            fn prctl(option: libc::c_int, arg2: libc::c_ulong,
                     arg3: libc::c_ulong, arg4: libc::c_ulong,
                     arg5: libc::c_ulong) -> libc::c_int;
        }
        const PR_SET_NAME: libc::c_int = 15;
        let cname = CString::new(name).unwrap_or_else(|_| {
            panic!("thread name may not contain interior null bytes")
        });
        unsafe {
            prctl(PR_SET_NAME, cname.as_ptr() as libc::c_ulong, 0, 0, 0);
        }
    }

    #[cfg(any(target_os = "freebsd",
              target_os = "dragonfly",
              target_os = "bitrig",
              target_os = "openbsd"))]
    pub fn set_name(name: &str) {
        extern {
            fn pthread_set_name_np(tid: libc::pthread_t,
                                   name: *const libc::c_char);
        }
        let cname = CString::new(name).unwrap();
        unsafe {
            pthread_set_name_np(pthread_self(), cname.as_ptr());
        }
    }

    #[cfg(any(target_os = "macos", target_os = "ios"))]
    pub fn set_name(name: &str) {
        extern {
            fn pthread_setname_np(name: *const libc::c_char) -> libc::c_int;
        }
        let cname = CString::new(name).unwrap();
        unsafe {
            pthread_setname_np(cname.as_ptr());
        }
    }

    #[cfg(target_os = "netbsd")]
    pub fn set_name(name: &str) {
        extern {
            fn pthread_setname_np(thread: libc::pthread_t,
                                  name: *const libc::c_char,
                                  arg: *mut libc::c_void) -> libc::c_int;
        }
        let cname = CString::new(&b"%s"[..]).unwrap();
        let carg = CString::new(name).unwrap();
        unsafe {
            pthread_setname_np(pthread_self(), cname.as_ptr(),
                               carg.as_ptr() as *mut libc::c_void);
        }
    }

    pub fn sleep(dur: Duration) {
        let mut ts = libc::timespec {
            tv_sec: dur.as_secs() as libc::time_t,
            tv_nsec: dur.subsec_nanos() as libc::c_long,
        };

        // If we're awoken with a signal then the return value will be -1 and
        // nanosleep will fill in `ts` with the remaining time.
        unsafe {
            while libc::nanosleep(&ts, &mut ts) == -1 {
                assert_eq!(os::errno(), libc::EINTR);
            }
        }
    }

    pub fn join(self) {
        unsafe {
            let ret = pthread_join(self.id, ptr::null_mut());
            mem::forget(self);
            debug_assert_eq!(ret, 0);
        }
    }

    pub fn id(&self) -> u32 {
        self.id as u32
    }
}

impl Drop for Thread {
    fn drop(&mut self) {
        let ret = unsafe { pthread_detach(self.id) };
        debug_assert_eq!(ret, 0);
    }
}

#[cfg(all(not(target_os = "linux"),
          not(target_os = "macos"),
          not(target_os = "bitrig"),
          not(all(target_os = "netbsd", not(target_vendor = "rumprun"))),
          not(target_os = "openbsd")))]
pub mod guard {
    pub unsafe fn current() -> Option<usize> { None }
    pub unsafe fn init() -> Option<usize> { None }
}


#[cfg(any(target_os = "linux",
          target_os = "macos",
          target_os = "bitrig",
          all(target_os = "netbsd", not(target_vendor = "rumprun")),
          target_os = "openbsd"))]
#[allow(unused_imports)]
pub mod guard {
    use prelude::v1::*;

    use libc::{self, pthread_t};
    use libc::funcs::posix88::mman::mmap;
    use libc::consts::os::posix88::{PROT_NONE,
                                    MAP_PRIVATE,
                                    MAP_ANON,
                                    MAP_FAILED,
                                    MAP_FIXED};
    use mem;
    use ptr;
    use super::{pthread_self, pthread_attr_destroy};
    use sys::os;

    #[cfg(any(target_os = "macos",
              target_os = "bitrig",
              target_os = "openbsd"))]
    unsafe fn get_stack_start() -> Option<*mut libc::c_void> {
        current().map(|s| s as *mut libc::c_void)
    }

    #[cfg(any(target_os = "linux", target_os = "android", target_os = "netbsd"))]
    unsafe fn get_stack_start() -> Option<*mut libc::c_void> {
        use super::pthread_attr_init;

        let mut ret = None;
        let mut attr: libc::pthread_attr_t = mem::zeroed();
        assert_eq!(pthread_attr_init(&mut attr), 0);
        if pthread_getattr_np(pthread_self(), &mut attr) == 0 {
            let mut stackaddr = ptr::null_mut();
            let mut stacksize = 0;
            assert_eq!(pthread_attr_getstack(&attr, &mut stackaddr,
                                             &mut stacksize), 0);
            ret = Some(stackaddr);
        }
        assert_eq!(pthread_attr_destroy(&mut attr), 0);
        ret
    }

    pub unsafe fn init() -> Option<usize> {
        let psize = os::page_size();
        let mut stackaddr = match get_stack_start() {
            Some(addr) => addr,
            None => return None,
        };

        // Ensure stackaddr is page aligned! A parent process might
        // have reset RLIMIT_STACK to be non-page aligned. The
        // pthread_attr_getstack() reports the usable stack area
        // stackaddr < stackaddr + stacksize, so if stackaddr is not
        // page-aligned, calculate the fix such that stackaddr <
        // new_page_aligned_stackaddr < stackaddr + stacksize
        let remainder = (stackaddr as usize) % psize;
        if remainder != 0 {
            stackaddr = ((stackaddr as usize) + psize - remainder)
                as *mut libc::c_void;
        }

        // Rellocate the last page of the stack.
        // This ensures SIGBUS will be raised on
        // stack overflow.
        let result = mmap(stackaddr,
                          psize as libc::size_t,
                          PROT_NONE,
                          MAP_PRIVATE | MAP_ANON | MAP_FIXED,
                          -1,
                          0);

        if result != stackaddr || result == MAP_FAILED {
            panic!("failed to allocate a guard page");
        }

        let offset = if cfg!(target_os = "linux") {2} else {1};

        Some(stackaddr as usize + offset * psize)
    }

    #[cfg(target_os = "macos")]
    pub unsafe fn current() -> Option<usize> {
        extern {
            fn pthread_get_stackaddr_np(thread: pthread_t) -> *mut libc::c_void;
            fn pthread_get_stacksize_np(thread: pthread_t) -> libc::size_t;
        }
        Some((pthread_get_stackaddr_np(pthread_self()) as libc::size_t -
              pthread_get_stacksize_np(pthread_self())) as usize)
    }

    #[cfg(any(target_os = "openbsd", target_os = "bitrig"))]
    pub unsafe fn current() -> Option<usize> {
        #[repr(C)]
        struct stack_t {
            ss_sp: *mut libc::c_void,
            ss_size: libc::size_t,
            ss_flags: libc::c_int,
        }
        extern {
            fn pthread_main_np() -> libc::c_uint;
            fn pthread_stackseg_np(thread: pthread_t,
                                   sinfo: *mut stack_t) -> libc::c_uint;
        }

        let mut current_stack: stack_t = mem::zeroed();
        assert_eq!(pthread_stackseg_np(pthread_self(), &mut current_stack), 0);

        let extra = if cfg!(target_os = "bitrig") {3} else {1} * os::page_size();
        Some(if pthread_main_np() == 1 {
            // main thread
            current_stack.ss_sp as usize - current_stack.ss_size as usize + extra
        } else {
            // new thread
            current_stack.ss_sp as usize - current_stack.ss_size as usize
        })
    }

    #[cfg(any(target_os = "linux", target_os = "android", target_os = "netbsd"))]
    pub unsafe fn current() -> Option<usize> {
        use super::pthread_attr_init;

        let mut ret = None;
        let mut attr: libc::pthread_attr_t = mem::zeroed();
        assert_eq!(pthread_attr_init(&mut attr), 0);
        if pthread_getattr_np(pthread_self(), &mut attr) == 0 {
            let mut guardsize = 0;
            assert_eq!(pthread_attr_getguardsize(&attr, &mut guardsize), 0);
            if guardsize == 0 {
                panic!("there is no guard page");
            }
            let mut stackaddr = ptr::null_mut();
            let mut size = 0;
            assert_eq!(pthread_attr_getstack(&attr, &mut stackaddr, &mut size), 0);

            ret = if cfg!(target_os = "netbsd") {
                Some(stackaddr as usize)
            } else {
                Some(stackaddr as usize + guardsize as usize)
            };
        }
        assert_eq!(pthread_attr_destroy(&mut attr), 0);
        ret
    }

    #[cfg(any(target_os = "linux", target_os = "android", target_os = "netbsd"))]
    extern {
        fn pthread_getattr_np(native: libc::pthread_t,
                              attr: *mut libc::pthread_attr_t) -> libc::c_int;
        fn pthread_attr_getguardsize(attr: *const libc::pthread_attr_t,
                                     guardsize: *mut libc::size_t) -> libc::c_int;
        fn pthread_attr_getstack(attr: *const libc::pthread_attr_t,
                                 stackaddr: *mut *mut libc::c_void,
                                 stacksize: *mut libc::size_t) -> libc::c_int;
    }
}

// glibc >= 2.15 has a __pthread_get_minstack() function that returns
// PTHREAD_STACK_MIN plus however many bytes are needed for thread-local
// storage.  We need that information to avoid blowing up when a small stack
// is created in an application with big thread-local storage requirements.
// See #6233 for rationale and details.
//
// Use dlsym to get the symbol value at runtime, both for
// compatibility with older versions of glibc, and to avoid creating
// dependencies on GLIBC_PRIVATE symbols.  Assumes that we've been
// dynamically linked to libpthread but that is currently always the
// case.  We previously used weak linkage (under the same assumption),
// but that caused Debian to detect an unnecessarily strict versioned
// dependency on libc6 (#23628).
#[cfg(target_os = "linux")]
#[allow(deprecated)]
fn min_stack_size(attr: *const libc::pthread_attr_t) -> usize {
    use dynamic_lib::DynamicLibrary;
    use sync::Once;

    type F = unsafe extern "C" fn(*const libc::pthread_attr_t) -> libc::size_t;
    static INIT: Once = Once::new();
    static mut __pthread_get_minstack: Option<F> = None;

    INIT.call_once(|| {
        let lib = match DynamicLibrary::open(None) {
            Ok(l) => l,
            Err(..) => return,
        };
        unsafe {
            if let Ok(f) = lib.symbol("__pthread_get_minstack") {
                __pthread_get_minstack = Some(mem::transmute::<*const (), F>(f));
            }
        }
    });

    match unsafe { __pthread_get_minstack } {
        None => PTHREAD_STACK_MIN as usize,
        Some(f) => unsafe { f(attr) as usize },
    }
}

// No point in looking up __pthread_get_minstack() on non-glibc
// platforms.
#[cfg(not(target_os = "linux"))]
fn min_stack_size(_: *const libc::pthread_attr_t) -> usize {
    PTHREAD_STACK_MIN as usize
}

extern {
    fn pthread_self() -> libc::pthread_t;
    fn pthread_create(native: *mut libc::pthread_t,
                      attr: *const libc::pthread_attr_t,
                      f: extern fn(*mut libc::c_void) -> *mut libc::c_void,
                      value: *mut libc::c_void) -> libc::c_int;
    fn pthread_join(native: libc::pthread_t,
                    value: *mut *mut libc::c_void) -> libc::c_int;
    fn pthread_attr_init(attr: *mut libc::pthread_attr_t) -> libc::c_int;
    fn pthread_attr_destroy(attr: *mut libc::pthread_attr_t) -> libc::c_int;
    fn pthread_attr_setstacksize(attr: *mut libc::pthread_attr_t,
                                 stack_size: libc::size_t) -> libc::c_int;
    fn pthread_attr_setdetachstate(attr: *mut libc::pthread_attr_t,
                                   state: libc::c_int) -> libc::c_int;
    fn pthread_detach(thread: libc::pthread_t) -> libc::c_int;
    fn sched_yield() -> libc::c_int;
}
