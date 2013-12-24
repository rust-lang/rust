// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.


// Implementation of Rust stack unwinding
//
// For background on exception handling and stack unwinding please see "Exception Handling in LLVM"
// (llvm.org/docs/ExceptionHandling.html) and documents linked from it.
// These are also good reads:
//     http://theofilos.cs.columbia.edu/blog/2013/09/22/base_abi/
//     http://monoinfinito.wordpress.com/series/exception-handling-in-c/
//     http://www.airs.com/blog/index.php?s=exception+frames
//
// ~~~ A brief summary ~~~
// Exception handling happens in two phases: a search phase and a cleanup phase.
//
// In both phases the unwinder walks stack frames from top to bottom using information from
// the stack frame unwind sections of the current process's modules ("module" here refers to
// an OS module, i.e. an executable or a dynamic library).
//
// For each stack frame, it invokes the associated "personality routine", whose address is also
// stored in the unwind info section.
//
// In the search phase, the job of a personality routine is to examine exception object being
// thrown, and to decide whether it should be caught at that stack frame.  Once the handler frame
// has been identified, cleanup phase begins.
//
// In the cleanup phase, personality routines invoke cleanup code associated with their
// stack frames (i.e. destructors).  Once stack has been unwound down to the handler frame level,
// unwinding stops and the last personality routine transfers control to its' catch block.
//
// ~~~ Frame unwind info registration ~~~
// Each module has its' own frame unwind info section (usually ".eh_frame"), and unwinder needs
// to know about all of them in order for unwinding to be able to cross module boundaries.
//
// On some platforms, like Linux, this is achieved by dynamically enumerating currently loaded
// modules via the dl_iterate_phdr() API and finding all .eh_frame sections.
//
// Others, like Windows, require modules to actively register their unwind info sections by calling
// __register_frame_info() API at startup.
// In the latter case it is essential that there is only one copy of the unwinder runtime
// in the process.  This is usually achieved by linking to the dynamic version of the unwind
// runtime.
//
// Currently Rust uses unwind runtime provided by libgcc.

use prelude::*;
use cast::transmute;
use task::TaskResult;
use libc::{c_void, c_int};
use self::libunwind::*;

mod libunwind {
    //! Unwind library interface

    #[allow(non_camel_case_types)];

    use libc::{uintptr_t, uint64_t};

    #[repr(C)]
    pub enum _Unwind_Action
    {
        _UA_SEARCH_PHASE = 1,
        _UA_CLEANUP_PHASE = 2,
        _UA_HANDLER_FRAME = 4,
        _UA_FORCE_UNWIND = 8,
        _UA_END_OF_STACK = 16,
    }

    #[repr(C)]
    pub enum _Unwind_Reason_Code {
        _URC_NO_REASON = 0,
        _URC_FOREIGN_EXCEPTION_CAUGHT = 1,
        _URC_FATAL_PHASE2_ERROR = 2,
        _URC_FATAL_PHASE1_ERROR = 3,
        _URC_NORMAL_STOP = 4,
        _URC_END_OF_STACK = 5,
        _URC_HANDLER_FOUND = 6,
        _URC_INSTALL_CONTEXT = 7,
        _URC_CONTINUE_UNWIND = 8,
    }

    pub type _Unwind_Exception_Class = uint64_t;

    pub type _Unwind_Word = uintptr_t;

    pub struct _Unwind_Exception {
        exception_class: _Unwind_Exception_Class,
        exception_cleanup: _Unwind_Exception_Cleanup_Fn,
        private_1: _Unwind_Word,
        private_2: _Unwind_Word,
    }

    pub enum _Unwind_Context {}

    pub type _Unwind_Exception_Cleanup_Fn = extern "C" fn(unwind_code: _Unwind_Reason_Code,
                                                          exception: *_Unwind_Exception);

    extern "C" {
        pub fn _Unwind_RaiseException(exception: *_Unwind_Exception) -> _Unwind_Reason_Code;
        pub fn _Unwind_DeleteException(exception: *_Unwind_Exception);
    }
}

pub struct Unwinder {
    unwinding: bool,
    cause: Option<~Any>
}

impl Unwinder {

    pub fn try(&mut self, f: ||) {
        use unstable::raw::Closure;

        unsafe {
            let closure: Closure = transmute(f);
            let code = transmute(closure.code);
            let env = transmute(closure.env);

            let ep = rust_try(try_fn, code, env);
            if !ep.is_null() {
                rtdebug!("Caught {}", (*ep).exception_class);
                _Unwind_DeleteException(ep);
            }
        }

        extern fn try_fn(code: *c_void, env: *c_void) {
            unsafe {
                let closure: Closure = Closure {
                    code: transmute(code),
                    env: transmute(env),
                };
                let closure: || = transmute(closure);
                closure();
            }
        }

        extern {
            // Rust's try-catch
            // When f(...) returns normally, the return value is null.
            // When f(...) throws, the return value is a pointer to the caught exception object.
            fn rust_try(f: extern "C" fn(*c_void, *c_void),
                        code: *c_void,
                        data: *c_void) -> *_Unwind_Exception;
        }
    }

    pub fn begin_unwind(&mut self, cause: ~Any) -> ! {
        rtdebug!("begin_unwind()");

        self.unwinding = true;
        self.cause = Some(cause);

        unsafe {
            let exception = ~_Unwind_Exception {
                exception_class: rust_exception_class(),
                exception_cleanup: exception_cleanup,
                private_1: 0,
                private_2: 0
            };
            let error = _Unwind_RaiseException(transmute(exception));
            rtabort!("Could not unwind stack, error = {}", error as int)
        }

        extern "C" fn exception_cleanup(_unwind_code: _Unwind_Reason_Code,
                                        exception: *_Unwind_Exception) {
            rtdebug!("exception_cleanup()");
            unsafe {
                let _: ~_Unwind_Exception = transmute(exception);
            }
        }
    }

    pub fn result(&mut self) -> TaskResult {
        if self.unwinding {
            Err(self.cause.take().unwrap())
        } else {
            Ok(())
        }
    }
}

// Rust's exception class identifier.  This is used by personality routines to
// determine whether the exception was thrown by their own runtime.
fn rust_exception_class() -> _Unwind_Exception_Class {
    let bytes = bytes!("MOZ\0RUST"); // vendor, language
    unsafe {
        let ptr: *_Unwind_Exception_Class = transmute(bytes.as_ptr());
        *ptr
    }
}


// We could implement our personality routine in pure Rust, however exception info decoding
// is tedious.  More importantly, personality routines have to handle various platform
// quirks, which are not fun to maintain.  For this reason, we attempt to reuse personality
// routine of the C language: __gcc_personality_v0.
//
// Since C does not support exception catching, __gcc_personality_v0 simply always
// returns _URC_CONTINUE_UNWIND in search phase, and always returns _URC_INSTALL_CONTEXT
// (i.e. "invoke cleanup code") in cleanup phase.
//
// This is pretty close to Rust's exception handling approach, except that Rust does have
// a single "catch-all" handler at the bottom of each task's stack.
// So we have two versions:
// - rust_eh_personality, used by all cleanup landing pads, which never catches, so
//   the behavior of __gcc_personality_v0 is perfectly adequate there, and
// - rust_eh_personality_catch, used only by rust_try(), which always catches.  This is
//   achieved by overriding the return value in search phase to always say "catch!".

extern "C" {
    fn __gcc_personality_v0(version: c_int,
                            actions: _Unwind_Action,
                            exception_class: _Unwind_Exception_Class,
                            ue_header: *_Unwind_Exception,
                            context: *_Unwind_Context) -> _Unwind_Reason_Code;
}

#[lang="eh_personality"]
#[no_mangle] // so we can reference it by name from middle/trans/base.rs
#[doc(hidden)]
#[cfg(not(test))]
pub extern "C" fn rust_eh_personality(version: c_int,
                                      actions: _Unwind_Action,
                                      exception_class: _Unwind_Exception_Class,
                                      ue_header: *_Unwind_Exception,
                                      context: *_Unwind_Context) -> _Unwind_Reason_Code {
    unsafe {
        __gcc_personality_v0(version, actions, exception_class, ue_header, context)
    }
}

#[no_mangle] // referenced from rust_try.ll
#[doc(hidden)]
#[cfg(not(test))]
pub extern "C" fn rust_eh_personality_catch(version: c_int,
                                            actions: _Unwind_Action,
                                            exception_class: _Unwind_Exception_Class,
                                            ue_header: *_Unwind_Exception,
                                            context: *_Unwind_Context) -> _Unwind_Reason_Code {
    if (actions as c_int & _UA_SEARCH_PHASE as c_int) != 0 { // search phase
        _URC_HANDLER_FOUND // catch!
    }
    else { // cleanup phase
        unsafe {
             __gcc_personality_v0(version, actions, exception_class, ue_header, context)
        }
    }
}
