// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use core::prelude::*;

use boxed::Box;
use cmp;
use mem;
use ptr;
use libc::consts::os::posix01::{PTHREAD_CREATE_JOINABLE, PTHREAD_STACK_MIN};
use libc;
use thunk::Thunk;

use sys_common::stack::RED_ZONE;
use sys_common::thread::*;

pub type rust_thread = libc::pthread_t;
pub type rust_thread_return = *mut u8;
pub type StartFn = extern "C" fn(*mut libc::c_void) -> rust_thread_return;

#[no_stack_check]
pub extern fn thread_start(main: *mut libc::c_void) -> rust_thread_return {
    return start_thread(main);
}

#[cfg(all(not(target_os = "linux"), not(target_os = "macos")))]
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

#[cfg(any(target_os = "linux", target_os = "macos"))]
pub mod guard {
    use super::*;
    #[cfg(any(target_os = "linux", target_os = "android"))]
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

    #[cfg(target_os = "macos")]
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

        let stackaddr = get_stack_start();

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

pub unsafe fn create(stack: uint, p: Thunk) -> rust_thread {
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
        panic!("failed to spawn native thread: {}", ret);
    }
    native
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

#[cfg(any(target_os = "linux"))]
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

#[cfg(target_os = "macos")]
extern {
    pub fn pthread_self() -> libc::pthread_t;
    pub fn pthread_get_stackaddr_np(thread: libc::pthread_t) -> *mut libc::c_void;
    pub fn pthread_get_stacksize_np(thread: libc::pthread_t) -> libc::size_t;
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
