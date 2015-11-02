// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use libc::{self, LONG};
use panicking::report_overflow;
use sys::windows::c;

pub struct Handler;

impl Handler {
    pub unsafe fn new() -> Handler {
        // This API isn't available on XP, so don't panic in that case and just
        // pray it works out ok.
        if c::SetThreadStackGuarantee(&mut 0x5000) == 0 {
            if libc::GetLastError() as u32 != libc::ERROR_CALL_NOT_IMPLEMENTED as u32 {
                panic!("failed to reserve stack space for exception handling");
            }
        }
        Handler
    }
}

extern "system" fn vectored_handler(ExceptionInfo: *mut c::EXCEPTION_POINTERS)
                                    -> LONG {
    unsafe {
        let rec = &(*(*ExceptionInfo).ExceptionRecord);
        let code = rec.ExceptionCode;

        if code == c::EXCEPTION_STACK_OVERFLOW {
            report_overflow();
        }
        c::EXCEPTION_CONTINUE_SEARCH
    }
}

pub unsafe fn init() {
    if c::AddVectoredExceptionHandler(0, vectored_handler).is_null() {
        panic!("failed to install exception handler");
    }
    // Set the thread stack guarantee for the main thread.
    let _h = Handler::new();
}

pub unsafe fn cleanup() {}
