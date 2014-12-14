// Copyright 2013-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Native os-thread management
//!
//! This modules contains bindings necessary for managing OS-level threads.
//! These functions operate outside of the rust runtime, creating threads
//! which are not used for scheduling in any way.

#![allow(non_camel_case_types)]

use core::prelude::*;

use alloc::boxed::Box;
use core::mem;
use core::uint;
use libc;
use thunk::{Thunk};

use stack;
use stack_overflow;

pub unsafe fn init() {
    imp::guard::init();
    stack_overflow::init();
}

pub unsafe fn cleanup() {
    stack_overflow::cleanup();
}

#[cfg(target_os = "windows")]
type StartFn = extern "system" fn(*mut libc::c_void) -> imp::rust_thread_return;

#[cfg(not(target_os = "windows"))]
type StartFn = extern "C" fn(*mut libc::c_void) -> imp::rust_thread_return;

/// This struct represents a native thread's state. This is used to join on an
/// existing thread created in the join-able state.
pub struct Thread<T> {
    native: imp::rust_thread,
    joined: bool,
    packet: Box<Option<T>>,
}

static DEFAULT_STACK_SIZE: uint = 1024 * 1024;

// This is the starting point of rust os threads. The first thing we do
// is make sure that we don't trigger __morestack (also why this has a
// no_stack_check annotation), and then we extract the main function
// and invoke it.
#[no_stack_check]
fn start_thread(main: *mut libc::c_void) -> imp::rust_thread_return {
    unsafe {
        stack::record_os_managed_stack_bounds(0, uint::MAX);
        let handler = stack_overflow::Handler::new();
        let f: Box<Thunk> = mem::transmute(main);
        f.invoke(());
        drop(handler);
        mem::transmute(0 as imp::rust_thread_return)
    }
}

#[no_stack_check]
#[cfg(target_os = "windows")]
extern "system" fn thread_start(main: *mut libc::c_void) -> imp::rust_thread_return {
    return start_thread(main);
}

#[no_stack_check]
#[cfg(not(target_os = "windows"))]
extern fn thread_start(main: *mut libc::c_void) -> imp::rust_thread_return {
    return start_thread(main);
}

/// Returns the last writable byte of the main thread's stack next to the guard
/// page. Must be called from the main thread.
pub fn main_guard_page() -> uint {
    unsafe {
        imp::guard::main()
    }
}

/// Returns the last writable byte of the current thread's stack next to the
/// guard page. Must not be called from the main thread.
pub fn current_guard_page() -> uint {
    unsafe {
        imp::guard::current()
    }
}

// There are two impl blocks b/c if T were specified at the top then it's just a
// pain to specify a type parameter on Thread::spawn (which doesn't need the
// type parameter).
impl Thread<()> {

    /// Starts execution of a new OS thread.
    ///
    /// This function will not wait for the thread to join, but a handle to the
    /// thread will be returned.
    ///
    /// Note that the handle returned is used to acquire the return value of the
    /// procedure `main`. The `join` function will wait for the thread to finish
    /// and return the value that `main` generated.
    ///
    /// Also note that the `Thread` returned will *always* wait for the thread
    /// to finish executing. This means that even if `join` is not explicitly
    /// called, when the `Thread` falls out of scope its destructor will block
    /// waiting for the OS thread.
    pub fn start<T,F>(main: F) -> Thread<T>
        where T:Send, F:FnOnce() -> T, F:Send
    {
        Thread::start_stack(DEFAULT_STACK_SIZE, main)
    }

    /// Performs the same functionality as `start`, but specifies an explicit
    /// stack size for the new thread.
    pub fn start_stack<T, F>(stack: uint, main: F) -> Thread<T>
        where T:Send, F:FnOnce() -> T, F:Send
    {
        // We need the address of the packet to fill in to be stable so when
        // `main` fills it in it's still valid, so allocate an extra box to do
        // so.
        let packet = box None;
        let packet2: *mut Option<T> = unsafe {
            *mem::transmute::<&Box<Option<T>>, *const *mut Option<T>>(&packet)
        };
        let native = unsafe {
            imp::create(stack, Thunk::new(move |:| {
                *packet2 = Some(main.call_once(()));
            }))
        };

        Thread {
            native: native,
            joined: false,
            packet: packet,
        }
    }

    /// This will spawn a new thread, but it will not wait for the thread to
    /// finish, nor is it possible to wait for the thread to finish.
    ///
    /// This corresponds to creating threads in the 'detached' state on unix
    /// systems. Note that platforms may not keep the main program alive even if
    /// there are detached thread still running around.
    pub fn spawn<F>(main: F)
        where F : FnOnce() + Send
    {
        Thread::spawn_stack(DEFAULT_STACK_SIZE, main)
    }

    /// Performs the same functionality as `spawn`, but explicitly specifies a
    /// stack size for the new thread.
    pub fn spawn_stack<F>(stack: uint, main: F)
        where F : FnOnce() + Send
    {
        unsafe {
            let handle = imp::create(stack, Thunk::new(main));
            imp::detach(handle);
        }
    }

    /// Relinquishes the CPU slot that this OS-thread is currently using,
    /// allowing another thread to run for awhile.
    pub fn yield_now() {
        unsafe { imp::yield_now(); }
    }
}

impl<T: Send> Thread<T> {
    /// Wait for this thread to finish, returning the result of the thread's
    /// calculation.
    pub fn join(mut self) -> T {
        assert!(!self.joined);
        unsafe { imp::join(self.native) };
        self.joined = true;
        assert!(self.packet.is_some());
        self.packet.take().unwrap()
    }
}

#[unsafe_destructor]
impl<T: Send> Drop for Thread<T> {
    fn drop(&mut self) {
        // This is required for correctness. If this is not done then the thread
        // would fill in a return box which no longer exists.
        if !self.joined {
            unsafe { imp::join(self.native) };
        }
    }
}

#[cfg(windows)]
#[allow(non_snake_case)]
mod imp {
    use alloc::boxed::Box;
    use core::cmp;
    use core::mem;
    use core::ptr;
    use libc;
    use libc::types::os::arch::extra::{LPSECURITY_ATTRIBUTES, SIZE_T, BOOL,
                                       LPVOID, DWORD, LPDWORD, HANDLE};
    use stack::RED_ZONE;
    use thunk::Thunk;

    pub type rust_thread = HANDLE;
    pub type rust_thread_return = DWORD;

    pub mod guard {
        pub unsafe fn main() -> uint {
            0
        }

        pub unsafe fn current() -> uint {
            0
        }

        pub unsafe fn init() {
        }
    }

    pub unsafe fn create(stack: uint, p: Thunk) -> rust_thread {
        let arg: *mut libc::c_void = mem::transmute(box p);

        // FIXME On UNIX, we guard against stack sizes that are too small but
        // that's because pthreads enforces that stacks are at least
        // PTHREAD_STACK_MIN bytes big.  Windows has no such lower limit, it's
        // just that below a certain threshold you can't do anything useful.
        // That threshold is application and architecture-specific, however.
        // For now, the only requirement is that it's big enough to hold the
        // red zone.  Round up to the next 64 kB because that's what the NT
        // kernel does, might as well make it explicit.  With the current
        // 20 kB red zone, that makes for a 64 kB minimum stack.
        let stack_size = (cmp::max(stack, RED_ZONE) + 0xfffe) & (-0xfffe - 1);
        let ret = CreateThread(ptr::null_mut(), stack_size as libc::size_t,
                               super::thread_start, arg, 0, ptr::null_mut());

        if ret as uint == 0 {
            // be sure to not leak the closure
            let _p: Box<Thunk> = mem::transmute(arg);
            panic!("failed to spawn native thread: {}", ret);
        }
        return ret;
    }

    pub unsafe fn join(native: rust_thread) {
        use libc::consts::os::extra::INFINITE;
        WaitForSingleObject(native, INFINITE);
    }

    pub unsafe fn detach(native: rust_thread) {
        assert!(libc::CloseHandle(native) != 0);
    }

    pub unsafe fn yield_now() {
        // This function will return 0 if there are no other threads to execute,
        // but this also means that the yield was useless so this isn't really a
        // case that needs to be worried about.
        SwitchToThread();
    }

    #[allow(non_snake_case)]
    extern "system" {
        fn CreateThread(lpThreadAttributes: LPSECURITY_ATTRIBUTES,
                        dwStackSize: SIZE_T,
                        lpStartAddress: super::StartFn,
                        lpParameter: LPVOID,
                        dwCreationFlags: DWORD,
                        lpThreadId: LPDWORD) -> HANDLE;
        fn WaitForSingleObject(hHandle: HANDLE, dwMilliseconds: DWORD) -> DWORD;
        fn SwitchToThread() -> BOOL;
    }
}

#[cfg(unix)]
mod imp {
    use core::prelude::*;

    use alloc::boxed::Box;
    use core::cmp;
    use core::mem;
    use core::ptr;
    use libc::consts::os::posix01::{PTHREAD_CREATE_JOINABLE, PTHREAD_STACK_MIN};
    use libc;
    use thunk::Thunk;

    use stack::RED_ZONE;

    pub type rust_thread = libc::pthread_t;
    pub type rust_thread_return = *mut u8;

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
        use core::mem;
        #[cfg(any(target_os = "linux", target_os = "android"))]
        use core::ptr;
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
        let ret = pthread_create(&mut native, &attr, super::thread_start, arg);
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
                          f: super::StartFn,
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
}

#[cfg(test)]
mod tests {
    use super::Thread;

    #[test]
    fn smoke() { Thread::start(move|| {}).join(); }

    #[test]
    fn data() { assert_eq!(Thread::start(move|| { 1i }).join(), 1); }

    #[test]
    fn detached() { Thread::spawn(move|| {}) }

    #[test]
    fn small_stacks() {
        assert_eq!(42i, Thread::start_stack(0, move|| 42i).join());
        assert_eq!(42i, Thread::start_stack(1, move|| 42i).join());
    }
}
