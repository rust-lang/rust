// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! # Implementation of Rust stack unwinding
//!
//! Also might be known as itanium exceptions, libunwind-based unwinding,
//! libgcc unwinding, etc. Unwinding on Unix currently always happens through
//! the libunwind support library, although we don't always link to libunwind
//! directly as it's often bundled directly into libgcc (or libgcc_s).
//!
//! For background on exception handling and stack unwinding please see
//! "Exception Handling in LLVM" (llvm.org/docs/ExceptionHandling.html) and
//! documents linked from it.
//!
//! These are also good reads:
//!
//! * http://mentorembedded.github.io/cxx-abi/abi-eh.html
//! * http://monoinfinito.wordpress.com/series/exception-handling-in-c/
//! * http://www.airs.com/blog/index.php?s=exception+frames
//!
//! The basic premise here is that we're going to literally throw an exception,
//! only with library support rather than language support. This interacts with
//! LLVM by using the `invoke` instruction all over the place instead of a
//! typical `call` instruction. As a reminder, when LLVM translates an `invoke`
//! instruction it requires that the surrounding function is attached with a
//! **personality** function. The major purpose of this module is to define
//! these personality functions.
//!
//! First, though, let's take a look at how unwinding works:
//!
//! ## A brief summary
//!
//! Exception handling happens in two phases: a search phase and a cleanup
//! phase.
//!
//! In both phases the unwinder walks stack frames from top to bottom using
//! information from the stack frame unwind sections of the current process's
//! modules ("module" here refers to an OS module, i.e. an executable or a
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
//!
//! ## `eh_personality` and `eh_personality_catch`
//!
//! The main personality function, `eh_personality`, is used by almost all Rust
//! functions that are translated. This personality indicates that no exceptions
//! should be caught, but cleanups should always be run. The second personality
//! function, `eh_personality_catch`, is distinct in that it indicates that
//! exceptions should be caught (no cleanups are run). The compiler will
//! annotate all generated functions with these two personalities, and then
//! we're just left to implement them over here!
//!
//! We could implement our personality routine in pure Rust, however exception
//! info decoding is tedious. More importantly, personality routines have to
//! handle various platform quirks, which are not fun to maintain. For this
//! reason, we attempt to reuse personality routine of the C language. This
//! comes under a number of names and ABIs, including `__gcc_personality_v0` and
//! `__gcc_personality_sj0`.
//!
//! Since C does not support exception catching, this personality function
//! simply always returns `_URC_CONTINUE_UNWIND` in search phase, and always
//! returns `_URC_INSTALL_CONTEXT` (i.e. "invoke cleanup code") in cleanup
//! phase.
//!
//! To implement the `eh_personality_catch` function, however, we detect when
//! the search phase is occurring and return `_URC_HANDLER_FOUND` to indicate
//! that we want to catch the exception.
//!
//! See also: `rustc_trans::trans::intrinsic::trans_gnu_try`

#![allow(private_no_mangle_fns)]

use prelude::v1::*;

use any::Any;
use sys::libunwind as uw;

#[repr(C)]
struct Exception {
    _uwe: uw::_Unwind_Exception,
    cause: Option<Box<Any + Send + 'static>>,
}

pub unsafe fn panic(data: Box<Any + Send + 'static>) -> ! {
    let exception: Box<_> = box Exception {
        _uwe: uw::_Unwind_Exception {
            exception_class: rust_exception_class(),
            exception_cleanup: exception_cleanup,
            private: [0; uw::unwinder_private_data_size],
        },
        cause: Some(data),
    };
    let exception_param = Box::into_raw(exception) as *mut uw::_Unwind_Exception;
    let error = uw::_Unwind_RaiseException(exception_param);
    rtabort!("Could not unwind stack, error = {}", error as isize);

    extern fn exception_cleanup(_unwind_code: uw::_Unwind_Reason_Code,
                                exception: *mut uw::_Unwind_Exception) {
        unsafe {
            let _: Box<Exception> = Box::from_raw(exception as *mut Exception);
        }
    }
}

#[cfg(not(stage0))]
pub fn payload() -> *mut u8 {
    0 as *mut u8
}

pub unsafe fn cleanup(ptr: *mut u8) -> Box<Any + Send + 'static> {
    let my_ep = ptr as *mut Exception;
    let cause = (*my_ep).cause.take();
    uw::_Unwind_DeleteException(ptr as *mut _);
    cause.unwrap()
}

// Rust's exception class identifier.  This is used by personality routines to
// determine whether the exception was thrown by their own runtime.
fn rust_exception_class() -> uw::_Unwind_Exception_Class {
    // M O Z \0  R U S T -- vendor, language
    0x4d4f5a_00_52555354
}

#[cfg(all(not(target_arch = "arm"), not(test)))]
pub mod eabi {
    use sys::libunwind as uw;
    use libc::c_int;

    extern {
        fn __gcc_personality_v0(version: c_int,
                                actions: uw::_Unwind_Action,
                                exception_class: uw::_Unwind_Exception_Class,
                                ue_header: *mut uw::_Unwind_Exception,
                                context: *mut uw::_Unwind_Context)
            -> uw::_Unwind_Reason_Code;
    }

    #[lang = "eh_personality"]
    #[no_mangle]
    extern fn rust_eh_personality(
        version: c_int,
        actions: uw::_Unwind_Action,
        exception_class: uw::_Unwind_Exception_Class,
        ue_header: *mut uw::_Unwind_Exception,
        context: *mut uw::_Unwind_Context
    ) -> uw::_Unwind_Reason_Code
    {
        unsafe {
            __gcc_personality_v0(version, actions, exception_class, ue_header,
                                 context)
        }
    }

    #[lang = "eh_personality_catch"]
    #[no_mangle]
    pub extern fn rust_eh_personality_catch(
        version: c_int,
        actions: uw::_Unwind_Action,
        exception_class: uw::_Unwind_Exception_Class,
        ue_header: *mut uw::_Unwind_Exception,
        context: *mut uw::_Unwind_Context
    ) -> uw::_Unwind_Reason_Code
    {

        if (actions as c_int & uw::_UA_SEARCH_PHASE as c_int) != 0 { // search phase
            uw::_URC_HANDLER_FOUND // catch!
        }
        else { // cleanup phase
            unsafe {
                __gcc_personality_v0(version, actions, exception_class, ue_header,
                                     context)
            }
        }
    }
}

// iOS on armv7 is using SjLj exceptions and therefore requires to use
// a specialized personality routine: __gcc_personality_sj0

#[cfg(all(target_os = "ios", target_arch = "arm", not(test)))]
pub mod eabi {
    use sys::libunwind as uw;
    use libc::c_int;

    extern {
        fn __gcc_personality_sj0(version: c_int,
                                actions: uw::_Unwind_Action,
                                exception_class: uw::_Unwind_Exception_Class,
                                ue_header: *mut uw::_Unwind_Exception,
                                context: *mut uw::_Unwind_Context)
            -> uw::_Unwind_Reason_Code;
    }

    #[lang = "eh_personality"]
    #[no_mangle]
    pub extern fn rust_eh_personality(
        version: c_int,
        actions: uw::_Unwind_Action,
        exception_class: uw::_Unwind_Exception_Class,
        ue_header: *mut uw::_Unwind_Exception,
        context: *mut uw::_Unwind_Context
    ) -> uw::_Unwind_Reason_Code
    {
        unsafe {
            __gcc_personality_sj0(version, actions, exception_class, ue_header,
                                  context)
        }
    }

    #[lang = "eh_personality_catch"]
    #[no_mangle]
    pub extern fn rust_eh_personality_catch(
        version: c_int,
        actions: uw::_Unwind_Action,
        exception_class: uw::_Unwind_Exception_Class,
        ue_header: *mut uw::_Unwind_Exception,
        context: *mut uw::_Unwind_Context
    ) -> uw::_Unwind_Reason_Code
    {
        if (actions as c_int & uw::_UA_SEARCH_PHASE as c_int) != 0 { // search phase
            uw::_URC_HANDLER_FOUND // catch!
        }
        else { // cleanup phase
            unsafe {
                __gcc_personality_sj0(version, actions, exception_class, ue_header,
                                      context)
            }
        }
    }
}


// ARM EHABI uses a slightly different personality routine signature,
// but otherwise works the same.
#[cfg(all(target_arch = "arm", not(target_os = "ios"), not(test)))]
pub mod eabi {
    use sys::libunwind as uw;
    use libc::c_int;

    extern {
        fn __gcc_personality_v0(state: uw::_Unwind_State,
                                ue_header: *mut uw::_Unwind_Exception,
                                context: *mut uw::_Unwind_Context)
            -> uw::_Unwind_Reason_Code;
    }

    #[lang = "eh_personality"]
    #[no_mangle]
    extern fn rust_eh_personality(
        state: uw::_Unwind_State,
        ue_header: *mut uw::_Unwind_Exception,
        context: *mut uw::_Unwind_Context
    ) -> uw::_Unwind_Reason_Code
    {
        unsafe {
            __gcc_personality_v0(state, ue_header, context)
        }
    }

    #[lang = "eh_personality_catch"]
    #[no_mangle]
    pub extern fn rust_eh_personality_catch(
        state: uw::_Unwind_State,
        ue_header: *mut uw::_Unwind_Exception,
        context: *mut uw::_Unwind_Context
    ) -> uw::_Unwind_Reason_Code
    {
        if (state as c_int & uw::_US_ACTION_MASK as c_int)
                           == uw::_US_VIRTUAL_UNWIND_FRAME as c_int { // search phase
            uw::_URC_HANDLER_FOUND // catch!
        }
        else { // cleanup phase
            unsafe {
                __gcc_personality_v0(state, ue_header, context)
            }
        }
    }
}
