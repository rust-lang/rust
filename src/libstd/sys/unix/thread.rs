// Copyright 2014-2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use core::prelude::*;

use io;
use boxed::Box;
use cmp;
use mem;
use ptr;
use libc::consts::os::posix01::{PTHREAD_CREATE_JOINABLE, PTHREAD_STACK_MIN};
use libc;
use thunk::Thunk;
use ffi::CString;

use sys_common::stack::RED_ZONE;
use sys_common::thread::*;

pub type rust_thread = libc::pthread_t;
pub type rust_thread_return = *mut u8;
pub type StartFn = extern "C" fn(*mut libc::c_void) -> rust_thread_return;

#[no_stack_check]
pub extern fn thread_start(main: *mut libc::c_void) -> rust_thread_return {
    return start_thread(main);
}

#[cfg(all(not(target_os = "linux"),
          not(target_os = "macos"),
          not(target_os = "openbsd")))]
pub mod guard {
    pub unsafe fn current() -> uint {
        0
    }

    pub unsafe fn main() -> uint {
        0
    }

    pub unsafe fn init() {
    }
}


#[cfg(any(target_os = "linux",
          target_os = "macos",
          target_os = "openbsd"))]
pub mod guard {
    use super::*;
    #[cfg(any(target_os = "linux",
              target_os = "android",
              target_os = "openbsd"))]
    use mem;
    #[cfg(any(target_os = "linux", target_os = "android"))]
    use ptr;
    use libc;
    use libc::funcs::posix88::mman::{mmap};
    use libc::consts::os::posix88::{PROT_NONE,
                                    MAP_PRIVATE,
                                    MAP_ANON,
                                    MAP_FAILED,
                                    MAP_FIXED};

    // These are initialized in init() and only read from after
    static mut PAGE_SIZE: uint = 0;
    static mut GUARD_PAGE: uint = 0;

    #[cfg(any(target_os = "macos", target_os = "openbsd"))]
    unsafe fn get_stack_start() -> *mut libc::c_void {
        current() as *mut libc::c_void
    }

    #[cfg(any(target_os = "linux", target_os = "android"))]
    unsafe fn get_stack_start() -> *mut libc::c_void {
        let mut attr: libc::pthread_attr_t = mem::zeroed();
        if pthread_getattr_np(pthread_self(), &mut attr) != 0 {
            panic!("failed to get thread attributes");
        }
        let mut stackaddr = ptr::null_mut();
        let mut stacksize = 0;
        if pthread_attr_getstack(&attr, &mut stackaddr, &mut stacksize) != 0 {
            panic!("failed to get stack information");
        }
        if pthread_attr_destroy(&mut attr) != 0 {
            panic!("failed to destroy thread attributes");
        }
        stackaddr
    }

    pub unsafe fn init() {
        let psize = libc::sysconf(libc::consts::os::sysconf::_SC_PAGESIZE);
        if psize == -1 {
            panic!("failed to get page size");
        }

        PAGE_SIZE = psize as uint;

        let mut stackaddr = get_stack_start();

        // Ensure stackaddr is page aligned! A parent process might
        // have reset RLIMIT_STACK to be non-page aligned. The
        // pthread_attr_getstack() reports the usable stack area
        // stackaddr < stackaddr + stacksize, so if stackaddr is not
        // page-aligned, calculate the fix such that stackaddr <
        // new_page_aligned_stackaddr < stackaddr + stacksize
        let remainder = (stackaddr as usize) % (PAGE_SIZE as usize);
        if remainder != 0 {
            stackaddr = ((stackaddr as usize) + (PAGE_SIZE as usize) - remainder)
                as *mut libc::c_void;
        }

        // Rellocate the last page of the stack.
        // This ensures SIGBUS will be raised on
        // stack overflow.
        let result = mmap(stackaddr,
                          PAGE_SIZE as libc::size_t,
                          PROT_NONE,
                          MAP_PRIVATE | MAP_ANON | MAP_FIXED,
                          -1,
                          0);

        if result != stackaddr || result == MAP_FAILED {
            panic!("failed to allocate a guard page");
        }

        let offset = if cfg!(target_os = "linux") {
            2
        } else {
            1
        };

        GUARD_PAGE = stackaddr as uint + offset * PAGE_SIZE;
    }

    pub unsafe fn main() -> uint {
        GUARD_PAGE
    }

    #[cfg(target_os = "macos")]
    pub unsafe fn current() -> uint {
        (pthread_get_stackaddr_np(pthread_self()) as libc::size_t -
         pthread_get_stacksize_np(pthread_self())) as uint
    }

    #[cfg(target_os = "openbsd")]
    pub unsafe fn current() -> uint {
        let mut current_stack: stack_t = mem::zeroed();
        if pthread_stackseg_np(pthread_self(), &mut current_stack) != 0 {
            panic!("failed to get current stack: pthread_stackseg_np")
        }

        if pthread_main_np() == 1 {
            // main thread
            current_stack.ss_sp as uint - current_stack.ss_size as uint + 3 * PAGE_SIZE as uint

        } else {
            // new thread
            current_stack.ss_sp as uint - current_stack.ss_size as uint
        }
    }

    #[cfg(any(target_os = "linux", target_os = "android"))]
    pub unsafe fn current() -> uint {
        let mut attr: libc::pthread_attr_t = mem::zeroed();
        if pthread_getattr_np(pthread_self(), &mut attr) != 0 {
            panic!("failed to get thread attributes");
        }
        let mut guardsize = 0;
        if pthread_attr_getguardsize(&attr, &mut guardsize) != 0 {
            panic!("failed to get stack guard page");
        }
        if guardsize == 0 {
            panic!("there is no guard page");
        }
        let mut stackaddr = ptr::null_mut();
        let mut stacksize = 0;
        if pthread_attr_getstack(&attr, &mut stackaddr, &mut stacksize) != 0 {
            panic!("failed to get stack information");
        }
        if pthread_attr_destroy(&mut attr) != 0 {
            panic!("failed to destroy thread attributes");
        }

        stackaddr as uint + guardsize as uint
    }
}

pub unsafe fn create(stack: uint, p: Thunk) -> io::Result<rust_thread> {
    let mut native: libc::pthread_t = mem::zeroed();
    let mut attr: libc::pthread_attr_t = mem::zeroed();
    assert_eq!(pthread_attr_init(&mut attr), 0);
    assert_eq!(pthread_attr_setdetachstate(&mut attr,
                                           PTHREAD_CREATE_JOINABLE), 0);

    // Reserve room for the red zone, the runtime's stack of last resort.
    let stack_size = cmp::max(stack, RED_ZONE + min_stack_size(&attr) as uint);
    match pthread_attr_setstacksize(&mut attr, stack_size as libc::size_t) {
        0 => {
        },
        libc::EINVAL => {
            // EINVAL means |stack_size| is either too small or not a
            // multiple of the system page size.  Because it's definitely
            // >= PTHREAD_STACK_MIN, it must be an alignment issue.
            // Round up to the nearest page and try again.
            let page_size = libc::sysconf(libc::_SC_PAGESIZE) as uint;
            let stack_size = (stack_size + page_size - 1) &
                             (-(page_size as int - 1) as uint - 1);
            assert_eq!(pthread_attr_setstacksize(&mut attr, stack_size as libc::size_t), 0);
        },
        errno => {
            // This cannot really happen.
            panic!("pthread_attr_setstacksize() error: {}", errno);
        },
    };

    let arg: *mut libc::c_void = mem::transmute(box p); // must box since sizeof(p)=2*uint
    let ret = pthread_create(&mut native, &attr, thread_start, arg);
    assert_eq!(pthread_attr_destroy(&mut attr), 0);

    if ret != 0 {
        // be sure to not leak the closure
        let _p: Box<Box<FnOnce()+Send>> = mem::transmute(arg);
        Err(io::Error::from_os_error(ret))
    } else {
        Ok(native)
    }
}

#[cfg(any(target_os = "linux", target_os = "android"))]
pub unsafe fn set_name(name: &str) {
    // pthread_setname_np() since glibc 2.12
    // availability autodetected via weak linkage
    let cname = CString::new(name).unwrap();
    type F = unsafe extern "C" fn(libc::pthread_t, *const libc::c_char) -> libc::c_int;
    extern {
        #[linkage = "extern_weak"]
        static pthread_setname_np: *const ();
    }
    if !pthread_setname_np.is_null() {
        unsafe {
            mem::transmute::<*const (), F>(pthread_setname_np)(pthread_self(), cname.as_ptr());
        }
    }
}

#[cfg(any(target_os = "freebsd",
          target_os = "dragonfly",
          target_os = "openbsd"))]
pub unsafe fn set_name(name: &str) {
    // pthread_set_name_np() since almost forever on all BSDs
    let cname = CString::new(name).unwrap();
    pthread_set_name_np(pthread_self(), cname.as_ptr());
}

#[cfg(any(target_os = "macos", target_os = "ios"))]
pub unsafe fn set_name(name: &str) {
    // pthread_setname_np() since OS X 10.6 and iOS 3.2
    let cname = CString::new(name).unwrap();
    pthread_setname_np(cname.as_ptr());
}

pub unsafe fn join(native: rust_thread) {
    assert_eq!(pthread_join(native, ptr::null_mut()), 0);
}

pub unsafe fn detach(native: rust_thread) {
    assert_eq!(pthread_detach(native), 0);
}

pub unsafe fn yield_now() { assert_eq!(sched_yield(), 0); }
// glibc >= 2.15 has a __pthread_get_minstack() function that returns
// PTHREAD_STACK_MIN plus however many bytes are needed for thread-local
// storage.  We need that information to avoid blowing up when a small stack
// is created in an application with big thread-local storage requirements.
// See #6233 for rationale and details.
//
// Link weakly to the symbol for compatibility with older versions of glibc.
// Assumes that we've been dynamically linked to libpthread but that is
// currently always the case.  Note that you need to check that the symbol
// is non-null before calling it!
#[cfg(target_os = "linux")]
fn min_stack_size(attr: *const libc::pthread_attr_t) -> libc::size_t {
    type F = unsafe extern "C" fn(*const libc::pthread_attr_t) -> libc::size_t;
    extern {
        #[linkage = "extern_weak"]
        static __pthread_get_minstack: *const ();
    }
    if __pthread_get_minstack.is_null() {
        PTHREAD_STACK_MIN
    } else {
        unsafe { mem::transmute::<*const (), F>(__pthread_get_minstack)(attr) }
    }
}

// __pthread_get_minstack() is marked as weak but extern_weak linkage is
// not supported on OS X, hence this kludge...
#[cfg(not(target_os = "linux"))]
fn min_stack_size(_: *const libc::pthread_attr_t) -> libc::size_t {
    PTHREAD_STACK_MIN
}

#[cfg(any(target_os = "linux", target_os = "android"))]
extern {
    pub fn pthread_self() -> libc::pthread_t;
    pub fn pthread_getattr_np(native: libc::pthread_t,
                              attr: *mut libc::pthread_attr_t) -> libc::c_int;
    pub fn pthread_attr_getguardsize(attr: *const libc::pthread_attr_t,
                                     guardsize: *mut libc::size_t) -> libc::c_int;
    pub fn pthread_attr_getstack(attr: *const libc::pthread_attr_t,
                                 stackaddr: *mut *mut libc::c_void,
                                 stacksize: *mut libc::size_t) -> libc::c_int;
}

#[cfg(any(target_os = "freebsd",
          target_os = "dragonfly",
          target_os = "openbsd"))]
extern {
    pub fn pthread_self() -> libc::pthread_t;
    fn pthread_set_name_np(tid: libc::pthread_t, name: *const libc::c_char);
}

#[cfg(any(target_os = "macos", target_os = "ios"))]
extern {
    pub fn pthread_self() -> libc::pthread_t;
    pub fn pthread_get_stackaddr_np(thread: libc::pthread_t) -> *mut libc::c_void;
    pub fn pthread_get_stacksize_np(thread: libc::pthread_t) -> libc::size_t;
    fn pthread_setname_np(name: *const libc::c_char) -> libc::c_int;
}

#[cfg(target_os = "openbsd")]
extern {
        pub fn pthread_stackseg_np(thread: libc::pthread_t,
                                   sinfo: *mut stack_t) -> libc::c_uint;
        pub fn pthread_main_np() -> libc::c_uint;
}

#[cfg(target_os = "openbsd")]
#[repr(C)]
pub struct stack_t {
    pub ss_sp: *mut libc::c_void,
    pub ss_size: libc::size_t,
    pub ss_flags: libc::c_int,
}

extern {
    fn pthread_create(native: *mut libc::pthread_t,
                      attr: *const libc::pthread_attr_t,
                      f: StartFn,
                      value: *mut libc::c_void) -> libc::c_int;
    fn pthread_join(native: libc::pthread_t,
                    value: *mut *mut libc::c_void) -> libc::c_int;
    fn pthread_attr_init(attr: *mut libc::pthread_attr_t) -> libc::c_int;
    pub fn pthread_attr_destroy(attr: *mut libc::pthread_attr_t) -> libc::c_int;
    fn pthread_attr_setstacksize(attr: *mut libc::pthread_attr_t,
                                 stack_size: libc::size_t) -> libc::c_int;
    fn pthread_attr_setdetachstate(attr: *mut libc::pthread_attr_t,
                                   state: libc::c_int) -> libc::c_int;
    fn pthread_detach(thread: libc::pthread_t) -> libc::c_int;
    fn sched_yield() -> libc::c_int;
}
