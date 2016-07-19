// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Implementation of panics backed by libgcc/libunwind (in some form)
//!
//! For background on exception handling and stack unwinding please see
//! "Exception Handling in LLVM" (llvm.org/docs/ExceptionHandling.html) and
//! documents linked from it.
//! These are also good reads:
//!     http://mentorembedded.github.io/cxx-abi/abi-eh.html
//!     http://monoinfinito.wordpress.com/series/exception-handling-in-c/
//!     http://www.airs.com/blog/index.php?s=exception+frames
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
//! that stack frame.  Once the handler frame has been identified, cleanup phase
//! begins.
//!
//! In the cleanup phase, the unwinder invokes each personality routine again.
//! This time it decides which (if any) cleanup code needs to be run for
//! the current stack frame.  If so, the control is transferred to a special
//! branch in the function body, the "landing pad", which invokes destructors,
//! frees memory, etc.  At the end of the landing pad, control is transferred
//! back to the unwinder and unwinding resumes.
//!
//! Once stack has been unwound down to the handler frame level, unwinding stops
//! and the last personality routine transfers control to the catch block.
//!
//! ## `eh_personality` and `eh_unwind_resume`
//!
//! These language items are used by the compiler when generating unwind info.
//! The first one is the personality routine described above.  The second one
//! allows compilation target to customize the process of resuming unwind at the
//! end of the landing pads. `eh_unwind_resume` is used only if
//! `custom_unwind_resume` flag in the target options is set.

#![allow(private_no_mangle_fns)]

use core::any::Any;
use core::ptr;
use alloc::boxed::Box;

use unwind as uw;

#[repr(C)]
struct Exception {
    _uwe: uw::_Unwind_Exception,
    cause: Option<Box<Any + Send>>,
}

pub unsafe fn panic(data: Box<Any + Send>) -> u32 {
    let exception = Box::new(Exception {
        _uwe: uw::_Unwind_Exception {
            exception_class: rust_exception_class(),
            exception_cleanup: exception_cleanup,
            private: [0; uw::unwinder_private_data_size],
        },
        cause: Some(data),
    });
    let exception_param = Box::into_raw(exception) as *mut uw::_Unwind_Exception;
    return uw::_Unwind_RaiseException(exception_param) as u32;

    extern "C" fn exception_cleanup(_unwind_code: uw::_Unwind_Reason_Code,
                                    exception: *mut uw::_Unwind_Exception) {
        unsafe {
            let _: Box<Exception> = Box::from_raw(exception as *mut Exception);
        }
    }
}

pub fn payload() -> *mut u8 {
    ptr::null_mut()
}

pub unsafe fn cleanup(ptr: *mut u8) -> Box<Any + Send> {
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

// We could implement our personality routine in Rust, however exception
// info decoding is tedious.  More importantly, personality routines have to
// handle various platform quirks, which are not fun to maintain.  For this
// reason, we attempt to reuse personality routine of the C language:
// __gcc_personality_v0.
//
// Since C does not support exception catching, __gcc_personality_v0 simply
// always returns _URC_CONTINUE_UNWIND in search phase, and always returns
// _URC_INSTALL_CONTEXT (i.e. "invoke cleanup code") in cleanup phase.
//
// This is pretty close to Rust's exception handling approach, except that Rust
// does have a single "catch-all" handler at the bottom of each thread's stack.
// So we have two versions of the personality routine:
// - rust_eh_personality, used by all cleanup landing pads, which never catches,
//   so the behavior of __gcc_personality_v0 is perfectly adequate there, and
// - rust_eh_personality_catch, used only by rust_try(), which always catches.
//
// See also: rustc_trans::trans::intrinsic::trans_gnu_try

#[cfg(all(not(target_arch = "arm"),
          not(all(windows, target_arch = "x86_64"))))]
pub mod eabi {
    use unwind as uw;
    use libc::c_int;

    extern "C" {
        fn __gcc_personality_v0(version: c_int,
                                actions: uw::_Unwind_Action,
                                exception_class: uw::_Unwind_Exception_Class,
                                ue_header: *mut uw::_Unwind_Exception,
                                context: *mut uw::_Unwind_Context)
                                -> uw::_Unwind_Reason_Code;
    }

    #[lang = "eh_personality"]
    #[no_mangle]
    extern "C" fn rust_eh_personality(version: c_int,
                                      actions: uw::_Unwind_Action,
                                      exception_class: uw::_Unwind_Exception_Class,
                                      ue_header: *mut uw::_Unwind_Exception,
                                      context: *mut uw::_Unwind_Context)
                                      -> uw::_Unwind_Reason_Code {
        unsafe { __gcc_personality_v0(version, actions, exception_class, ue_header, context) }
    }

    #[lang = "eh_personality_catch"]
    #[no_mangle]
    pub extern "C" fn rust_eh_personality_catch(version: c_int,
                                                actions: uw::_Unwind_Action,
                                                exception_class: uw::_Unwind_Exception_Class,
                                                ue_header: *mut uw::_Unwind_Exception,
                                                context: *mut uw::_Unwind_Context)
                                                -> uw::_Unwind_Reason_Code {

        if (actions as c_int & uw::_UA_SEARCH_PHASE as c_int) != 0 {
            // search phase
            uw::_URC_HANDLER_FOUND // catch!
        } else {
            // cleanup phase
            unsafe { __gcc_personality_v0(version, actions, exception_class, ue_header, context) }
        }
    }
}

// iOS on armv7 is using SjLj exceptions and therefore requires to use
// a specialized personality routine: __gcc_personality_sj0

#[cfg(all(target_os = "ios", target_arch = "arm"))]
pub mod eabi {
    use unwind as uw;
    use libc::c_int;

    extern "C" {
        fn __gcc_personality_sj0(version: c_int,
                                 actions: uw::_Unwind_Action,
                                 exception_class: uw::_Unwind_Exception_Class,
                                 ue_header: *mut uw::_Unwind_Exception,
                                 context: *mut uw::_Unwind_Context)
                                 -> uw::_Unwind_Reason_Code;
    }

    #[lang = "eh_personality"]
    #[no_mangle]
    pub extern "C" fn rust_eh_personality(version: c_int,
                                          actions: uw::_Unwind_Action,
                                          exception_class: uw::_Unwind_Exception_Class,
                                          ue_header: *mut uw::_Unwind_Exception,
                                          context: *mut uw::_Unwind_Context)
                                          -> uw::_Unwind_Reason_Code {
        unsafe { __gcc_personality_sj0(version, actions, exception_class, ue_header, context) }
    }

    #[lang = "eh_personality_catch"]
    #[no_mangle]
    pub extern "C" fn rust_eh_personality_catch(version: c_int,
                                                actions: uw::_Unwind_Action,
                                                exception_class: uw::_Unwind_Exception_Class,
                                                ue_header: *mut uw::_Unwind_Exception,
                                                context: *mut uw::_Unwind_Context)
                                                -> uw::_Unwind_Reason_Code {
        if (actions as c_int & uw::_UA_SEARCH_PHASE as c_int) != 0 {
            // search phase
            uw::_URC_HANDLER_FOUND // catch!
        } else {
            // cleanup phase
            unsafe { __gcc_personality_sj0(version, actions, exception_class, ue_header, context) }
        }
    }
}


// ARM EHABI uses a slightly different personality routine signature,
// but otherwise works the same.
#[cfg(all(target_arch = "arm", not(target_os = "ios")))]
pub mod eabi {
    use unwind as uw;
    use libc::c_int;

    extern "C" {
        fn __gcc_personality_v0(state: uw::_Unwind_State,
                                ue_header: *mut uw::_Unwind_Exception,
                                context: *mut uw::_Unwind_Context)
                                -> uw::_Unwind_Reason_Code;
    }

    #[lang = "eh_personality"]
    #[no_mangle]
    extern "C" fn rust_eh_personality(state: uw::_Unwind_State,
                                      ue_header: *mut uw::_Unwind_Exception,
                                      context: *mut uw::_Unwind_Context)
                                      -> uw::_Unwind_Reason_Code {
        unsafe { __gcc_personality_v0(state, ue_header, context) }
    }

    #[lang = "eh_personality_catch"]
    #[no_mangle]
    pub extern "C" fn rust_eh_personality_catch(state: uw::_Unwind_State,
                                                ue_header: *mut uw::_Unwind_Exception,
                                                context: *mut uw::_Unwind_Context)
                                                -> uw::_Unwind_Reason_Code {
        // Backtraces on ARM will call the personality routine with
        // state == _US_VIRTUAL_UNWIND_FRAME | _US_FORCE_UNWIND. In those cases
        // we want to continue unwinding the stack, otherwise all our backtraces
        // would end at __rust_try.
        if (state as c_int & uw::_US_ACTION_MASK as c_int) ==
           uw::_US_VIRTUAL_UNWIND_FRAME as c_int &&
           (state as c_int & uw::_US_FORCE_UNWIND as c_int) == 0 {
            // search phase
            uw::_URC_HANDLER_FOUND // catch!
        } else {
            // cleanup phase
            unsafe { __gcc_personality_v0(state, ue_header, context) }
        }
    }
}

// See docs in the `unwind` module.
#[cfg(all(target_os="windows", target_arch = "x86", target_env="gnu"))]
#[lang = "eh_unwind_resume"]
#[unwind]
unsafe extern "C" fn rust_eh_unwind_resume(panic_ctx: *mut u8) -> ! {
    uw::_Unwind_Resume(panic_ctx as *mut uw::_Unwind_Exception);
}

// Frame unwind info registration
//
// Each module's image contains a frame unwind info section (usually
// ".eh_frame").  When a module is loaded/unloaded into the process, the
// unwinder must be informed about the location of this section in memory. The
// methods of achieving that vary by the platform.  On some (e.g. Linux), the
// unwinder can discover unwind info sections on its own (by dynamically
// enumerating currently loaded modules via the dl_iterate_phdr() API and
// finding their ".eh_frame" sections); Others, like Windows, require modules
// to actively register their unwind info sections via unwinder API.
//
// This module defines two symbols which are referenced and called from
// rsbegin.rs to reigster our information with the GCC runtime. The
// implementation of stack unwinding is (for now) deferred to libgcc_eh, however
// Rust crates use these Rust-specific entry points to avoid potential clashes
// with any GCC runtime.
#[cfg(all(target_os="windows", target_arch = "x86", target_env="gnu"))]
pub mod eh_frame_registry {
    #[link(name = "gcc_eh")]
    #[cfg(not(cargobuild))]
    extern "C" {}

    extern "C" {
        fn __register_frame_info(eh_frame_begin: *const u8, object: *mut u8);
        fn __deregister_frame_info(eh_frame_begin: *const u8, object: *mut u8);
    }

    #[no_mangle]
    pub unsafe extern "C" fn rust_eh_register_frames(eh_frame_begin: *const u8, object: *mut u8) {
        __register_frame_info(eh_frame_begin, object);
    }

    #[no_mangle]
    pub unsafe extern "C" fn rust_eh_unregister_frames(eh_frame_begin: *const u8,
                                                       object: *mut u8) {
        __deregister_frame_info(eh_frame_begin, object);
    }
}
