// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use libc;
use mem;
use ptr;
use sys::windows::c;
use sys::windows::handle::Handle;
use sys::error::{self, Result};
use sys::inner::*;
use time::Duration;

pub struct Thread {
    handle: Handle
}

pub unsafe fn new<'a>(stack: usize, f: unsafe extern fn(usize) -> usize, data: usize)
                      -> Result<Thread> {
    // FIXME On UNIX, we guard against stack sizes that are too small but
    // that's because pthreads enforces that stacks are at least
    // PTHREAD_STACK_MIN bytes big.  Windows has no such lower limit, it's
    // just that below a certain threshold you can't do anything useful.
    // That threshold is application and architecture-specific, however.
    // Round up to the next 64 kB because that's what the NT kernel does,
    // might as well make it explicit.
    let stack_size = (stack + 0xfffe) & (!0xfffe);
    let ret = c::CreateThread(ptr::null_mut(), stack_size as libc::size_t,
                              mem::transmute(f), data as *mut _,
                              0, ptr::null_mut());

    if ret as usize == 0 {
        error::expect_last_result()
    } else {
        Ok(Thread { handle: Handle::from_inner(ret) })
    }
}

pub fn set_name(_name: &str) -> Result<()> {
    // Windows threads are nameless
    // The names in MSVC debugger are obtained using a "magic" exception,
    // which requires a use of MS C++ extensions.
    // See https://msdn.microsoft.com/en-us/library/xcb2z8hs.aspx
    Ok(())
}

impl Thread {
    pub fn join(self) -> Result<()> {
        use libc::consts::os::extra::INFINITE;
        c::cvt_neg1(unsafe { c::WaitForSingleObject(*self.handle.as_inner(), INFINITE) as i32 }).map(drop)
    }
}

pub fn yield_() {
    // This function will return 0 if there are no other threads to execute,
    // but this also means that the yield was useless so this isn't really a
    // case that needs to be worried about.
    unsafe { c::SwitchToThread(); }
}

pub fn sleep(dur: Duration) -> Result<()> {
    unsafe {
        c::Sleep(c::dur2timeout(dur));
    }
    Ok(())
}
