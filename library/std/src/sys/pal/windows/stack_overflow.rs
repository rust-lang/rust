#![cfg_attr(test, allow(dead_code))]

use crate::sys::c;
use crate::thread;

/// Reserve stack space for use in stack overflow exceptions.
pub unsafe fn reserve_stack() {
    let result = c::SetThreadStackGuarantee(&mut 0x5000);
    // Reserving stack space is not critical so we allow it to fail in the released build of libstd.
    // We still use debug assert here so that CI will test that we haven't made a mistake calling the function.
    debug_assert_ne!(result, 0, "failed to reserve stack space for exception handling");
}

unsafe extern "system" fn vectored_handler(ExceptionInfo: *mut c::EXCEPTION_POINTERS) -> c::LONG {
    unsafe {
        let rec = &(*(*ExceptionInfo).ExceptionRecord);
        let code = rec.ExceptionCode;

        if code == c::EXCEPTION_STACK_OVERFLOW {
            rtprintpanic!(
                "\nthread '{}' has overflowed its stack\n",
                thread::current().name().unwrap_or("<unknown>")
            );
        }
        c::EXCEPTION_CONTINUE_SEARCH
    }
}

pub unsafe fn init() {
    let result = c::AddVectoredExceptionHandler(0, Some(vectored_handler));
    // Similar to the above, adding the stack overflow handler is allowed to fail
    // but a debug assert is used so CI will still test that it normally works.
    debug_assert!(!result.is_null(), "failed to install exception handler");
    // Set the thread stack guarantee for the main thread.
    reserve_stack();
}
