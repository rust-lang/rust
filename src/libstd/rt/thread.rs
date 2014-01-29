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

#[allow(non_camel_case_types)];

use cast;
use kinds::Send;
use libc;
use ops::Drop;
use option::{Option, Some, None};
use uint;

type StartFn = extern "C" fn(*libc::c_void) -> imp::rust_thread_return;

/// This struct represents a native thread's state. This is used to join on an
/// existing thread created in the join-able state.
pub struct Thread<T> {
    priv native: imp::rust_thread,
    priv joined: bool,
    priv packet: ~Option<T>,
}

static DEFAULT_STACK_SIZE: uint = 1024 * 1024;

// This is the starting point of rust os threads. The first thing we do
// is make sure that we don't trigger __morestack (also why this has a
// no_split_stack annotation), and then we extract the main function
// and invoke it.
#[no_split_stack]
extern fn thread_start(main: *libc::c_void) -> imp::rust_thread_return {
    use unstable::stack;
    unsafe {
        stack::record_stack_bounds(0, uint::MAX);
        let f: ~proc() = cast::transmute(main);
        (*f)();
        cast::transmute(0 as imp::rust_thread_return)
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
    pub fn start<T: Send>(main: proc() -> T) -> Thread<T> {
        Thread::start_stack(DEFAULT_STACK_SIZE, main)
    }

    /// Performs the same functionality as `start`, but specifies an explicit
    /// stack size for the new thread.
    pub fn start_stack<T: Send>(stack: uint, main: proc() -> T) -> Thread<T> {

        // We need the address of the packet to fill in to be stable so when
        // `main` fills it in it's still valid, so allocate an extra ~ box to do
        // so.
        let packet = ~None;
        let packet2: *mut Option<T> = unsafe {
            *cast::transmute::<&~Option<T>, **mut Option<T>>(&packet)
        };
        let main: proc() = proc() unsafe { *packet2 = Some(main()); };
        let native = unsafe { imp::create(stack, ~main) };

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
    pub fn spawn(main: proc()) {
        Thread::spawn_stack(DEFAULT_STACK_SIZE, main)
    }

    /// Performs the same functionality as `spawn`, but explicitly specifies a
    /// stack size for the new thread.
    pub fn spawn_stack(stack: uint, main: proc()) {
        unsafe {
            let handle = imp::create(stack, ~main);
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
        self.packet.take_unwrap()
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
mod imp {
    use cast;
    use cmp;
    use libc;
    use libc::types::os::arch::extra::{LPSECURITY_ATTRIBUTES, SIZE_T, BOOL,
                                       LPVOID, DWORD, LPDWORD, HANDLE};
    use ptr;
    use unstable::stack::RED_ZONE;

    pub type rust_thread = HANDLE;
    pub type rust_thread_return = DWORD;

    pub unsafe fn create(stack: uint, p: ~proc()) -> rust_thread {
        let arg: *mut libc::c_void = cast::transmute(p);
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
        CreateThread(ptr::mut_null(), stack_size as libc::size_t,
                     super::thread_start, arg, 0, ptr::mut_null())
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
    use cast;
    use cmp;
    use libc::consts::os::posix01::{PTHREAD_CREATE_JOINABLE, PTHREAD_STACK_MIN};
    use libc;
    use os;
    use ptr;
    use unstable::intrinsics;
    use unstable::stack::RED_ZONE;

    pub type rust_thread = libc::pthread_t;
    pub type rust_thread_return = *u8;

    pub unsafe fn create(stack: uint, p: ~proc()) -> rust_thread {
        let mut native: libc::pthread_t = intrinsics::uninit();
        let mut attr: libc::pthread_attr_t = intrinsics::uninit();
        assert_eq!(pthread_attr_init(&mut attr), 0);
        assert_eq!(pthread_attr_setdetachstate(&mut attr,
                                               PTHREAD_CREATE_JOINABLE), 0);

        // Reserve room for the red zone, the runtime's stack of last resort.
        let stack_size = cmp::max(stack, RED_ZONE + __pthread_get_minstack(&attr) as uint);
        match pthread_attr_setstacksize(&mut attr, stack_size as libc::size_t) {
            0 => {
            },
            libc::EINVAL => {
                // EINVAL means |stack_size| is either too small or not a
                // multiple of the system page size.  Because it's definitely
                // >= PTHREAD_STACK_MIN, it must be an alignment issue.
                // Round up to the neareast page and try again.
                let page_size = os::page_size();
                let stack_size = (stack_size + page_size - 1) & (-(page_size - 1) - 1);
                assert_eq!(pthread_attr_setstacksize(&mut attr, stack_size as libc::size_t), 0);
            },
            errno => {
                // This cannot really happen.
                fail!("pthread_attr_setstacksize() error: {} ({})", os::last_os_error(), errno);
            },
        };

        let arg: *libc::c_void = cast::transmute(p);
        assert_eq!(pthread_create(&mut native, &attr,
                                  super::thread_start, arg), 0);
        native
    }

    pub unsafe fn join(native: rust_thread) {
        assert_eq!(pthread_join(native, ptr::null()), 0);
    }

    pub unsafe fn detach(native: rust_thread) {
        assert_eq!(pthread_detach(native), 0);
    }

    #[cfg(target_os = "macos")]
    #[cfg(target_os = "android")]
    pub unsafe fn yield_now() { assert_eq!(sched_yield(), 0); }

    #[cfg(not(target_os = "macos"), not(target_os = "android"))]
    pub unsafe fn yield_now() { assert_eq!(pthread_yield(), 0); }

    #[cfg(not(target_os = "linux"))]
    unsafe fn __pthread_get_minstack(_: *libc::pthread_attr_t) -> libc::size_t {
        libc::PTHREAD_STACK_MIN
    }

    // glibc >= 2.15 has a __pthread_get_minstack() function that returns
    // PTHREAD_STACK_MIN plus however many bytes are needed for thread-local
    // storage.  We need that information to avoid blowing up when a small stack
    // is created in an application with big thread-local storage requirements.
    // See #6233 for rationale and details.
    //
    // Dynamically resolve the symbol for compatibility with older versions
    // of glibc.  Assumes that we've been dynamically linked to libpthread
    // but that is currently always the case.  Note that this means we take
    // a dlopen/dlsym/dlclose hit for every new thread.  Mitigating that by
    // caching the symbol or the function's return value has its drawbacks:
    //
    //  * Caching the symbol breaks when libpthread.so is reloaded because
    //    its address changes.
    //
    //  * Caching the return value assumes that it's a fixed quantity.
    //    Not very future-proof and untrue in the presence of guard pages
    //    The reason __pthread_get_minstack() takes a *libc::pthread_attr_t
    //    as its argument is because it takes pthread_attr_setguardsize() into
    //    account.
    //
    // A better solution is to define __pthread_get_minstack() as a weak symbol
    // but there is currently no way to express that in Rust code.
    #[cfg(target_os = "linux")]
    unsafe fn __pthread_get_minstack(attr: *libc::pthread_attr_t) -> libc::size_t {
        use option::None;
        use result::{Err, Ok};
        use unstable::dynamic_lib;
        match dynamic_lib::DynamicLibrary::open(None) {
            Err(err) => fail!("DynamicLibrary::open(): {}", err),
            Ok(handle) => {
                match handle.symbol::<extern "C" fn(*libc::pthread_attr_t) ->
                                     libc::size_t>("__pthread_get_minstack") {
                    Err(_) => libc::PTHREAD_STACK_MIN,
                    Ok(__pthread_get_minstack) => __pthread_get_minstack(attr),
                }
            }
        }
    }

    extern {
        fn pthread_create(native: *mut libc::pthread_t,
                          attr: *libc::pthread_attr_t,
                          f: super::StartFn,
                          value: *libc::c_void) -> libc::c_int;
        fn pthread_join(native: libc::pthread_t,
                        value: **libc::c_void) -> libc::c_int;
        fn pthread_attr_init(attr: *mut libc::pthread_attr_t) -> libc::c_int;
        fn pthread_attr_setstacksize(attr: *mut libc::pthread_attr_t,
                                     stack_size: libc::size_t) -> libc::c_int;
        fn pthread_attr_setdetachstate(attr: *mut libc::pthread_attr_t,
                                       state: libc::c_int) -> libc::c_int;
        fn pthread_detach(thread: libc::pthread_t) -> libc::c_int;

        #[cfg(target_os = "macos")]
        #[cfg(target_os = "android")]
        fn sched_yield() -> libc::c_int;
        #[cfg(not(target_os = "macos"), not(target_os = "android"))]
        fn pthread_yield() -> libc::c_int;
    }
}

#[cfg(test)]
mod tests {
    use super::Thread;

    #[test]
    fn smoke() { Thread::start(proc (){}).join(); }

    #[test]
    fn data() { assert_eq!(Thread::start(proc () { 1 }).join(), 1); }

    #[test]
    fn detached() { Thread::spawn(proc () {}) }

    #[test]
    fn small_stacks() {
        assert_eq!(42, Thread::start_stack(0, proc () 42).join());
        assert_eq!(42, Thread::start_stack(1, proc () 42).join());
    }
}
