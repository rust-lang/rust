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

use sys::error::{Result, Error};
use sys::thread_local::StaticOsKey;

use cmp;
#[cfg(not(target_env = "newlib"))]
use ffi::CString;
use libc::consts::os::posix01::PTHREAD_STACK_MIN;
use libc;
use mem;
use ptr;
use sys::unix::{c, cvt_r};
use time::Duration;

pub struct Thread {
    id: libc::pthread_t,
}

// Some platforms may have pthread_t as a pointer in which case we still want
// a thread to be Send/Sync
unsafe impl Send for Thread {}
unsafe impl Sync for Thread {}

pub unsafe fn new(stack: usize, f: unsafe extern fn(usize) -> usize, data: usize) -> Result<Thread> {
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
            let page_size = c::page_size();
            let stack_size = (stack_size + page_size - 1) &
                             (-(page_size as isize - 1) as usize - 1);
            let stack_size = stack_size as libc::size_t;
            assert_eq!(pthread_attr_setstacksize(&mut attr, stack_size), 0);
        }
    };

    let ret = pthread_create(&mut native, &attr, mem::transmute(f),
                             data as *mut _);
    assert_eq!(pthread_attr_destroy(&mut attr), 0);

    if ret != 0 {
        Err(Error::from_code(ret))
    } else {
        Ok(Thread { id: native })
    }
}

pub fn yield_() {
    let ret = unsafe { sched_yield() };
    debug_assert_eq!(ret, 0);
}

#[cfg(any(target_os = "linux", target_os = "android"))]
pub fn set_name(name: &str) -> Result<()> {
    // pthread wrapper only appeared in glibc 2.12, so we use syscall
    // directly.
    extern {
        fn prctl(option: libc::c_int, arg2: libc::c_ulong,
                 arg3: libc::c_ulong, arg4: libc::c_ulong,
                 arg5: libc::c_ulong) -> libc::c_int;
    }
    const PR_SET_NAME: libc::c_int = 15;
    let cname = try!(CString::new(name));
    unsafe {
        prctl(PR_SET_NAME, cname.as_ptr() as libc::c_ulong, 0, 0, 0);
    }

    Ok(())
}

#[cfg(any(target_os = "freebsd",
          target_os = "dragonfly",
          target_os = "bitrig",
          target_os = "openbsd"))]
pub fn set_name(name: &str) -> Result<()> {
    extern {
        fn pthread_set_name_np(tid: libc::pthread_t,
                               name: *const libc::c_char);
    }
    let cname = try!(CString::new(name));
    unsafe {
        pthread_set_name_np(pthread_self(), cname.as_ptr());
    }
    Ok(())
}

#[cfg(any(target_os = "macos", target_os = "ios"))]
pub fn set_name(name: &str) -> Result<()>  {
    extern {
        fn pthread_setname_np(name: *const libc::c_char) -> libc::c_int;
    }
    let cname = try!(CString::new(name));
    unsafe {
        pthread_setname_np(cname.as_ptr());
    }
    Ok(())
}

#[cfg(target_os = "netbsd")]
pub fn set_name(name: &str) -> Result<()>  {
    extern {
        fn pthread_setname_np(thread: libc::pthread_t,
                              name: *const libc::c_char,
                              arg: *mut libc::c_void) -> libc::c_int;
    }
    let cname = CString::new(&b"%s"[..]).unwrap();
    let carg = try!(CString::new(name));
    unsafe {
        pthread_setname_np(pthread_self(), cname.as_ptr(),
                           carg.as_ptr() as *mut libc::c_void);
    }
    Ok(())
}

#[cfg(target_env = "newlib")]
pub unsafe fn set_name(_name: &str) {
    // Newlib has no way to set a thread name.
    Ok(())
}

pub fn sleep(dur: Duration) -> Result<()> {
    let mut ts = libc::timespec {
        tv_sec: dur.as_secs() as libc::time_t,
        tv_nsec: dur.subsec_nanos() as libc::c_long,
    };

    // If we're awoken with a signal then the return value will be -1 and
    // nanosleep will fill in `ts` with the remaining time.
    cvt_r(|| unsafe { libc::nanosleep(&ts, &mut ts) }).map(drop)
}

impl Thread {
    pub fn join(self) -> Result<()> {
        unsafe {
            let ret = pthread_join(self.id, ptr::null_mut());
            mem::forget(self);
            if ret == 0 {
                Ok(())
            } else {
                Err(Error::from_code(ret))
            }
        }
    }
}

static THREAD_GUARD: StaticOsKey = StaticOsKey::new(None);

impl Thread {
    pub fn get_guard() -> usize {
        unsafe { THREAD_GUARD.get() as usize }
    }

    pub unsafe fn guard_current() {
        THREAD_GUARD.set(self::guard::current().unwrap_or(0) as *mut _)
    }

    pub unsafe fn guard_init() {
        THREAD_GUARD.set(self::guard::init().unwrap_or(0) as *mut _)
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
mod guard {
    pub unsafe fn current() -> Option<usize> { None }
    pub unsafe fn init() -> Option<usize> { None }
}


#[cfg(any(target_os = "linux",
          target_os = "macos",
          target_os = "bitrig",
          all(target_os = "netbsd", not(target_vendor = "rumprun")),
          target_os = "openbsd"))]
#[allow(unused_imports)]
mod guard {
    use libc::{self, pthread_t};
    use libc::funcs::posix88::mman::mmap;
    use libc::consts::os::posix88::{PROT_NONE,
                                    MAP_PRIVATE,
                                    MAP_ANON,
                                    MAP_FAILED,
                                    MAP_FIXED};
    use mem;
    use ptr;
    use sys::unix::c;
    use super::{pthread_self, pthread_attr_destroy};

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
        let psize = c::page_size();
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

        let extra = if cfg!(target_os = "bitrig") {3} else {1} * c::page_size();
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
    use sys::dynamic_lib as dl;
    use sync::Once;

    type F = unsafe extern "C" fn(*const libc::pthread_attr_t) -> libc::size_t;
    static INIT: Once = Once::new();
    static mut __pthread_get_minstack: Option<F> = None;

    INIT.call_once(|| {
        let lib = match dl::open(None) {
            Ok(l) => l,
            Err(..) => return,
        };
        unsafe {
            if let Ok(f) = lib.symbol("__pthread_get_minstack") {
                __pthread_get_minstack = Some(mem::transmute(f));
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
