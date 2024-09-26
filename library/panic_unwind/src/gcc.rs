//! Implementation of panics backed by libgcc/libunwind (in some form).
//!
//! For background on exception handling and stack unwinding please see
//! "Exception Handling in LLVM" (llvm.org/docs/ExceptionHandling.html) and
//! documents linked from it.
//! These are also good reads:
//!  * <https://itanium-cxx-abi.github.io/cxx-abi/abi-eh.html>
//!  * <https://monoinfinito.wordpress.com/series/exception-handling-in-c/>
//!  * <https://www.airs.com/blog/index.php?s=exception+frames>
//!
//! ## A brief summary
//!
//! Exception handling happens in two phases: a search phase and a cleanup
//! phase.
//!
//! In both phases the unwinder walks stack frames from top to bottom using
//! information from the stack frame unwind sections of the current process's
//! modules ("module" here refers to an OS module, i.e., an executable or a
//! dynamic library).
//!
//! For each stack frame, it invokes the associated "personality routine", whose
//! address is also stored in the unwind info section.
//!
//! In the search phase, the job of a personality routine is to examine
//! exception object being thrown, and to decide whether it should be caught at
//! that stack frame. Once the handler frame has been identified, cleanup phase
//! begins.
//!
//! In the cleanup phase, the unwinder invokes each personality routine again.
//! This time it decides which (if any) cleanup code needs to be run for
//! the current stack frame. If so, the control is transferred to a special
//! branch in the function body, the "landing pad", which invokes destructors,
//! frees memory, etc. At the end of the landing pad, control is transferred
//! back to the unwinder and unwinding resumes.
//!
//! Once stack has been unwound down to the handler frame level, unwinding stops
//! and the last personality routine transfers control to the catch block.

use alloc::boxed::Box;
use core::any::Any;
use core::ptr;

use unwind as uw;

// In case where multiple copies of std exist in a single process,
// we use address of this static variable to distinguish an exception raised by
// this copy and some other copy (which needs to be treated as foreign exception).
static CANARY: u8 = 0;

// NOTE(nbdd0121)
// There is an ABI stability requirement on this struct.
// The first two field must be `_Unwind_Exception` and `canary`,
// as it may be accessed by a different version of the std with a different compiler.
#[repr(C)]
struct Exception {
    _uwe: uw::_Unwind_Exception,
    canary: *const u8,
    cause: Box<dyn Any + Send>,
}

pub unsafe fn panic(data: Box<dyn Any + Send>) -> u32 {
    let exception = Box::new(Exception {
        _uwe: uw::_Unwind_Exception {
            exception_class: RUST_EXCEPTION_CLASS,
            exception_cleanup: Some(exception_cleanup),
            private: [core::ptr::null(); uw::unwinder_private_data_size],
        },
        canary: &CANARY,
        cause: data,
    });
    let exception_param = Box::into_raw(exception) as *mut uw::_Unwind_Exception;
    return uw::_Unwind_RaiseException(exception_param) as u32;

    extern "C" fn exception_cleanup(
        _unwind_code: uw::_Unwind_Reason_Code,
        exception: *mut uw::_Unwind_Exception,
    ) {
        unsafe {
            let _: Box<Exception> = Box::from_raw(exception as *mut Exception);
            super::__rust_drop_panic();
        }
    }
}

pub unsafe fn cleanup(ptr: *mut u8) -> Box<dyn Any + Send> {
    let exception = ptr as *mut uw::_Unwind_Exception;
    if (*exception).exception_class != RUST_EXCEPTION_CLASS {
        uw::_Unwind_DeleteException(exception);
        super::__rust_foreign_exception();
    }

    let exception = exception.cast::<Exception>();
    // Just access the canary field, avoid accessing the entire `Exception` as
    // it can be a foreign Rust exception.
    let canary = (&raw const (*exception).canary).read();
    if !ptr::eq(canary, &CANARY) {
        // A foreign Rust exception, treat it slightly differently from other
        // foreign exceptions, because call into `_Unwind_DeleteException` will
        // call into `__rust_drop_panic` which produces a confusing
        // "Rust panic must be rethrown" message.
        super::__rust_foreign_exception();
    }

    let exception = Box::from_raw(exception as *mut Exception);
    exception.cause
}

// Rust's exception class identifier.  This is used by personality routines to
// determine whether the exception was thrown by their own runtime.
const RUST_EXCEPTION_CLASS: uw::_Unwind_Exception_Class = u64::from_ne_bytes(*b"MOZ\0RUST");
