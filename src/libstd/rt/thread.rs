// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#[allow(non_camel_case_types)];

use cast;
use libc;
use ops::Drop;
use uint;
use ptr;

#[cfg(windows)]
use libc::types::os::arch::extra::{LPSECURITY_ATTRIBUTES, SIZE_T,
                                   LPVOID, DWORD, LPDWORD, HANDLE};

#[cfg(windows)] type rust_thread = HANDLE;
#[cfg(unix)] type rust_thread = libc::pthread_t;

pub struct Thread {
    priv native: rust_thread,
    priv joined: bool
}

static DEFAULT_STACK_SIZE: libc::size_t = 1024*1024;

#[cfg(windows)] type rust_thread_return = DWORD;
#[cfg(unix)] type rust_thread_return = *libc::c_void;

impl Thread {

    pub fn start(main: ~fn()) -> Thread {
        // This is the starting point of rust os threads. The first thing we do
        // is make sure that we don't trigger __morestack (also why this has a
        // no_split_stack annotation), and then we extract the main function
        // and invoke it.
        #[no_split_stack]
        extern "C" fn thread_start(trampoline: *libc::c_void) -> rust_thread_return {
            use rt::context;
            unsafe {
                context::record_stack_bounds(0, uint::max_value);
                let f: ~~fn() = cast::transmute(trampoline);
                (*f)();
            }
            unsafe { cast::transmute(0) }
        }

        let native = native_thread_create(thread_start, ~main);
        Thread {
            native: native,
            joined: false,
        }
    }

    pub fn join(mut self) {
        assert!(!self.joined);
        native_thread_join(self.native);
        self.joined = true;
    }
}

#[cfg(windows)]
fn native_thread_create(thread_start: extern "C" fn(*libc::c_void) -> rust_thread_return,
                        tramp: ~~fn()) -> rust_thread {
    #[fixed_stack_segment];

    unsafe {
        let ptr: *mut libc::c_void = cast::transmute(tramp);
        CreateThread(ptr::mut_null(), DEFAULT_STACK_SIZE, thread_start, ptr, 0, ptr::mut_null())
    }
}

#[cfg(windows)]
fn native_thread_join(native: rust_thread) {
    #[fixed_stack_segment];
    use libc::consts::os::extra::INFINITE;
    unsafe { WaitForSingleObject(native, INFINITE); }
}

#[cfg(unix)]
fn native_thread_create(thread_start: extern "C" fn(*libc::c_void) -> rust_thread_return,
                        tramp: ~~fn()) -> rust_thread {
    #[fixed_stack_segment];

    use unstable::intrinsics;
    let mut native: libc::pthread_t = unsafe { intrinsics::uninit() };

    unsafe {
        use libc::consts::os::posix01::PTHREAD_CREATE_JOINABLE;

        let mut attr: libc::pthread_attr_t = intrinsics::uninit();
        assert!(pthread_attr_init(&mut attr) == 0);
        assert!(pthread_attr_setstacksize(&mut attr, DEFAULT_STACK_SIZE) == 0);
        assert!(pthread_attr_setdetachstate(&mut attr, PTHREAD_CREATE_JOINABLE) == 0);

        let ptr: *libc::c_void = cast::transmute(tramp);
        assert!(pthread_create(&mut native, &attr, thread_start, ptr) == 0);
    }
    native
}

#[cfg(unix)]
fn native_thread_join(native: rust_thread) {
    #[fixed_stack_segment];
    unsafe { assert!(pthread_join(native, ptr::null()) == 0) }
}

impl Drop for Thread {
    fn drop(&mut self) {
        #[fixed_stack_segment]; #[inline(never)];
        assert!(self.joined);
    }
}

#[cfg(windows, target_arch = "x86")]
extern "stdcall" {
    fn CreateThread(lpThreadAttributes: LPSECURITY_ATTRIBUTES, dwStackSize: SIZE_T,
                    lpStartAddress: extern "C" fn(*libc::c_void) -> rust_thread_return,
                    lpParameter: LPVOID, dwCreationFlags: DWORD, lpThreadId: LPDWORD) -> HANDLE;
    fn WaitForSingleObject(hHandle: HANDLE, dwMilliseconds: DWORD) -> DWORD;
}

#[cfg(windows, target_arch = "x86_64")]
extern {
    fn CreateThread(lpThreadAttributes: LPSECURITY_ATTRIBUTES, dwStackSize: SIZE_T,
                    lpStartAddress: extern "C" fn(*libc::c_void) -> rust_thread_return,
                    lpParameter: LPVOID, dwCreationFlags: DWORD, lpThreadId: LPDWORD) -> HANDLE;
    fn WaitForSingleObject(hHandle: HANDLE, dwMilliseconds: DWORD) -> DWORD;
}

#[cfg(unix)]
extern {
    fn pthread_create(native: *mut libc::pthread_t, attr: *libc::pthread_attr_t,
                      f: extern "C" fn(*libc::c_void) -> rust_thread_return,
                      value: *libc::c_void) -> libc::c_int;
    fn pthread_join(native: libc::pthread_t, value: **libc::c_void) -> libc::c_int;
    fn pthread_attr_init(attr: *mut libc::pthread_attr_t) -> libc::c_int;
    fn pthread_attr_setstacksize(attr: *mut libc::pthread_attr_t,
                                 stack_size: libc::size_t) -> libc::c_int;
    fn pthread_attr_setdetachstate(attr: *mut libc::pthread_attr_t,
                                   state: libc::c_int) -> libc::c_int;
}
