#![cfg_attr(test, allow(dead_code))]

use crate::sys::c;
use crate::thread;

/// Reserve stack space for use in stack overflow exceptions.
pub fn reserve_stack() {
    let result = unsafe { c::SetThreadStackGuarantee(&mut 0x5000) };
    // Reserving stack space is not critical so we allow it to fail in the released build of libstd.
    // We still use debug assert here so that CI will test that we haven't made a mistake calling the function.
    debug_assert_ne!(result, 0, "failed to reserve stack space for exception handling");
}

unsafe extern "system" fn vectored_handler(ExceptionInfo: *mut c::EXCEPTION_POINTERS) -> i32 {
    // SAFETY: It's up to the caller (which in this case is the OS) to ensure that `ExceptionInfo` is valid.
    unsafe {
        let rec = &(*(*ExceptionInfo).ExceptionRecord);
        let code = rec.ExceptionCode;

        if code == c::EXCEPTION_STACK_OVERFLOW {
            thread::with_current_name(|name| {
                let name = name.unwrap_or("<unknown>");
                rtprintpanic!("\nthread '{name}' has overflowed its stack\n");
            });
        }
        c::EXCEPTION_CONTINUE_SEARCH
    }
}

pub fn init() {
    // SAFETY: `vectored_handler` has the correct ABI and is safe to call during exception handling.
    unsafe {
        let result = c::AddVectoredExceptionHandler(0, Some(vectored_handler));
        // Similar to the above, adding the stack overflow handler is allowed to fail
        // but a debug assert is used so CI will still test that it normally works.
        debug_assert!(!result.is_null(), "failed to install exception handler");
    }
    // Set the thread stack guarantee for the main thread.
    reserve_stack();
}
