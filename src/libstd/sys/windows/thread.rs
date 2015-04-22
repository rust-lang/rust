// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use prelude::v1::*;

use alloc::boxed::FnBox;
use cmp;
use io;
use libc::{self, c_void, DWORD};
use mem;
use ptr;
use sys::c;
use sys::handle::Handle;
use sys_common::stack::RED_ZONE;
use sys_common::thread::*;
use time::Duration;

pub struct Thread {
    handle: Handle
}

impl Thread {
    pub unsafe fn new<'a>(stack: usize, p: Box<FnBox() + 'a>)
                          -> io::Result<Thread> {
        let p = box p;

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
        let ret = c::CreateThread(ptr::null_mut(), stack_size as libc::size_t,
                                  thread_start, &*p as *const _ as *mut _,
                                  0, ptr::null_mut());

        return if ret as usize == 0 {
            Err(io::Error::last_os_error())
        } else {
            mem::forget(p); // ownership passed to CreateThread
            Ok(Thread { handle: Handle::new(ret) })
        };

        #[no_stack_check]
        extern "system" fn thread_start(main: *mut libc::c_void) -> DWORD {
            unsafe { start_thread(main); }
            0
        }
    }

    pub fn set_name(_name: &str) {
        // Windows threads are nameless
        // The names in MSVC debugger are obtained using a "magic" exception,
        // which requires a use of MS C++ extensions.
        // See https://msdn.microsoft.com/en-us/library/xcb2z8hs.aspx
    }

    pub fn join(self) {
        use libc::consts::os::extra::INFINITE;
        unsafe { c::WaitForSingleObject(self.handle.raw(), INFINITE); }
    }

    pub fn yield_now() {
        // This function will return 0 if there are no other threads to execute,
        // but this also means that the yield was useless so this isn't really a
        // case that needs to be worried about.
        unsafe { c::SwitchToThread(); }
    }

    pub fn sleep(dur: Duration) {
        unsafe {
            if dur < Duration::zero() {
                return Thread::yield_now()
            }
            let ms = dur.num_milliseconds();
            // if we have a fractional number of milliseconds then add an extra
            // millisecond to sleep for
            let extra = dur - Duration::milliseconds(ms);
            let ms = ms + if extra.is_zero() {0} else {1};
            c::Sleep(ms as DWORD);
        }
    }
}

pub mod guard {
    pub unsafe fn main() -> usize { 0 }
    pub unsafe fn current() -> usize { 0 }
    pub unsafe fn init() {}
}
