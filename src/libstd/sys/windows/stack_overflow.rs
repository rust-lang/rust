#![cfg_attr(test, allow(dead_code))]

use crate::sys_common::util::report_overflow;
use crate::sys::c;

pub struct Handler;

impl Handler {
    pub unsafe fn new() -> Handler {
        // This API isn't available on XP, so don't panic in that case and just
        // pray it works out ok.
        if c::SetThreadStackGuarantee(&mut 0x5000) == 0 {
            if c::GetLastError() as u32 != c::ERROR_CALL_NOT_IMPLEMENTED as u32 {
                panic!("failed to reserve stack space for exception handling");
            }
        }
        Handler
    }
}

extern "system" fn vectored_handler(ExceptionInfo: *mut c::EXCEPTION_POINTERS)
                                    -> c::LONG {
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
