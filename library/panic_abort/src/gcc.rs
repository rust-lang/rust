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

#![allow(nonstandard_style)]

use libc::c_int;

// The following code is based on GCC's C and C++ personality routines.  For reference, see:
// https://github.com/gcc-mirror/gcc/blob/master/libstdc++-v3/libsupc++/eh_personality.cc
// https://github.com/gcc-mirror/gcc/blob/trunk/libgcc/unwind-c.c

cfg_if::cfg_if! {
    if #[cfg(all(target_arch = "arm", not(target_os = "ios"), not(target_os = "netbsd")))] {
        // ARM EHABI personality routine.
        // https://infocenter.arm.com/help/topic/com.arm.doc.ihi0038b/IHI0038B_ehabi.pdf
        //
        // iOS uses the default routine instead since it uses SjLj unwinding.
        #[rustc_std_internal_symbol]
        unsafe extern "C" fn rust_eh_personality(_state: _Unwind_State,
                                                 _exception_object: *mut _Unwind_Exception,
                                                 _context: *mut _Unwind_Context)
                                                 -> _Unwind_Reason_Code {
            crate::do_abort()
        }
    } else {
        cfg_if::cfg_if! {
            if #[cfg(all(windows, target_arch = "x86_64", target_env = "gnu"))] {
                // On x86_64 MinGW targets, the unwinding mechanism is SEH however the unwind
                // handler data (aka LSDA) uses GCC-compatible encoding.
                #[rustc_std_internal_symbol]
                #[allow(nonstandard_style)]
                unsafe extern "C" fn rust_eh_personality(_exceptionRecord: *mut EXCEPTION_RECORD,
                        _establisherFrame: LPVOID,
                        _contextRecord: *mut CONTEXT,
                        _dispatcherContext: *mut DISPATCHER_CONTEXT)
                        -> EXCEPTION_DISPOSITION {
                    crate::do_abort();
                }
            } else {
                // The personality routine for most of our targets.
                #[rustc_std_internal_symbol]
                unsafe extern "C" fn rust_eh_personality(_version: c_int,
                        _actions: _Unwind_Action,
                        _exception_class: _Unwind_Exception_Class,
                        _exception_object: *mut _Unwind_Exception,
                        _context: *mut _Unwind_Context)
                        -> _Unwind_Reason_Code {
                    crate::do_abort();
                }
            }
        }
    }
}

// Frame unwind info registration
//
// Each module's image contains a frame unwind info section (usually
// ".eh_frame").  When a module is loaded/unloaded into the process, the
// unwinder must be informed about the location of this section in memory. The
// methods of achieving that vary by the platform.  On some (e.g., Linux), the
// unwinder can discover unwind info sections on its own (by dynamically
// enumerating currently loaded modules via the dl_iterate_phdr() API and
// finding their ".eh_frame" sections); Others, like Windows, require modules
// to actively register their unwind info sections via unwinder API.
//
// This module defines two symbols which are referenced and called from
// rsbegin.rs to register our information with the GCC runtime. The
// implementation of stack unwinding is (for now) deferred to libgcc_eh, however
// Rust crates use these Rust-specific entry points to avoid potential clashes
// with any GCC runtime.
#[cfg(all(target_os = "windows", target_arch = "x86", target_env = "gnu"))]
mod eh_frame_registry {
    extern "C" {
        fn __register_frame_info(eh_frame_begin: *const u8, object: *mut u8);
        fn __deregister_frame_info(eh_frame_begin: *const u8, object: *mut u8);
    }

    #[rustc_std_internal_symbol]
    unsafe extern "C" fn rust_eh_register_frames(eh_frame_begin: *const u8, object: *mut u8) {
        __register_frame_info(eh_frame_begin, object);
    }

    #[rustc_std_internal_symbol]
    unsafe extern "C" fn rust_eh_unregister_frames(eh_frame_begin: *const u8, object: *mut u8) {
        __deregister_frame_info(eh_frame_begin, object);
    }
}

use libc::{c_void, uintptr_t};

#[repr(C)]
#[derive(Debug, Copy, Clone, PartialEq)]
enum _Unwind_Reason_Code {
    _URC_NO_REASON = 0,
    _URC_FOREIGN_EXCEPTION_CAUGHT = 1,
    _URC_FATAL_PHASE2_ERROR = 2,
    _URC_FATAL_PHASE1_ERROR = 3,
    _URC_NORMAL_STOP = 4,
    _URC_END_OF_STACK = 5,
    _URC_HANDLER_FOUND = 6,
    _URC_INSTALL_CONTEXT = 7,
    _URC_CONTINUE_UNWIND = 8,
    _URC_FAILURE = 9, // used only by ARM EHABI
}

type _Unwind_Exception_Class = u64;
type _Unwind_Word = uintptr_t;
type _Unwind_Ptr = uintptr_t;
type _Unwind_Trace_Fn =
    extern "C" fn(ctx: *mut _Unwind_Context, arg: *mut c_void) -> _Unwind_Reason_Code;

enum _Unwind_Exception {}

enum _Unwind_Context {}

type _Unwind_Exception_Cleanup_Fn =
    extern "C" fn(unwind_code: _Unwind_Reason_Code, exception: *mut _Unwind_Exception);

cfg_if::cfg_if! {
if #[cfg(any(target_os = "ios", target_os = "netbsd", not(target_arch = "arm")))] {
    // Not ARM EHABI
    #[repr(C)]
    #[derive(Copy, Clone, PartialEq)]
    enum _Unwind_Action {
        _UA_SEARCH_PHASE = 1,
        _UA_CLEANUP_PHASE = 2,
        _UA_HANDLER_FRAME = 4,
        _UA_FORCE_UNWIND = 8,
        _UA_END_OF_STACK = 16,
    }

} else {
    // ARM EHABI
    #[repr(C)]
    #[derive(Copy, Clone, PartialEq)]
    enum _Unwind_State {
        _US_VIRTUAL_UNWIND_FRAME = 0,
        _US_UNWIND_FRAME_STARTING = 1,
        _US_UNWIND_FRAME_RESUME = 2,
        _US_ACTION_MASK = 3,
        _US_FORCE_UNWIND = 8,
        _US_END_OF_STACK = 16,
    }
}
} // cfg_if!

cfg_if::cfg_if! {
if #[cfg(all(windows, target_arch = "x86_64", target_env = "gnu"))] {
    // We declare these as opaque types. This is fine since you just need to
    // pass them to _GCC_specific_handler and forget about them.
    enum EXCEPTION_RECORD {}
    type LPVOID = *mut c_void;
    enum CONTEXT {}
    enum DISPATCHER_CONTEXT {}
    type EXCEPTION_DISPOSITION = c_int;
}
} // cfg_if!
