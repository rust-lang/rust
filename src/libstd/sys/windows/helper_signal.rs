// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use libc::{self, BOOL, LPCSTR, HANDLE, LPSECURITY_ATTRIBUTES, CloseHandle};
use ptr;

pub type signal = HANDLE;

pub fn new() -> (HANDLE, HANDLE) {
    unsafe {
        let handle = CreateEventA(ptr::null_mut(), libc::FALSE, libc::FALSE,
                                  ptr::null());
        (handle, handle)
    }
}

pub fn signal(handle: HANDLE) {
    assert!(unsafe { SetEvent(handle) != 0 });
}

pub fn close(handle: HANDLE) {
    assert!(unsafe { CloseHandle(handle) != 0 });
}

extern "system" {
    fn CreateEventA(lpSecurityAttributes: LPSECURITY_ATTRIBUTES,
                    bManualReset: BOOL,
                    bInitialState: BOOL,
                    lpName: LPCSTR) -> HANDLE;
    fn SetEvent(hEvent: HANDLE) -> BOOL;
}
