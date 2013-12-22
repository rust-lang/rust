// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use prelude::*;
use cast::transmute;
use task::TaskResult;
use libc::{abort, c_void, c_int};
use self::libunwind::*;

mod libunwind {
    //! Unwind library interface

    #[allow(non_camel_case_types)];

    use libc::{uintptr_t, uint64_t};

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

    #[repr(C)]
    pub enum _Unwind_Action
    {
        _UA_SEARCH_PHASE = 1,
        _UA_CLEANUP_PHASE = 2,
        _UA_HANDLER_FRAME = 4,
        _UA_FORCE_UNWIND = 8,
        _UA_END_OF_STACK = 16,
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
            error!("Could not unwind stack, error = {}", error as int)
            abort();
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

// We could implement the personality routine in pure Rust, however, __gcc_personality_v0
// (the "C" personality) already does everything we need in terms of invoking cleanup
// landing pads.  It also handles some system-dependent corner cases, which is nice.
// The only thing it doesn't do, is exception catching, but we can fix that (see below).
extern "C" {
    fn __gcc_personality_v0(version: c_int,
                            actions: _Unwind_Action,
                            exception_class: _Unwind_Exception_Class,
                            ue_header: *_Unwind_Exception,
                            context: *_Unwind_Context) -> _Unwind_Reason_Code;
}

// This variant is referenced from all cleanup landing pads
#[lang="eh_personality"]
#[doc(hidden)]
#[cfg(not(test))]
pub extern "C" fn eh_rust_personality(version: c_int,
                                      actions: _Unwind_Action,
                                      exception_class: _Unwind_Exception_Class,
                                      ue_header: *_Unwind_Exception,
                                      context: *_Unwind_Context) -> _Unwind_Reason_Code {
    unsafe {
        __gcc_personality_v0(version, actions, exception_class, ue_header, context)
    }
}

// And this one is used in one place only: rust_try() (in rt/rust_try.ll).
#[no_mangle]
#[doc(hidden)]
#[cfg(not(test))]
pub extern "C" fn eh_rust_personality_catch(version: c_int,
                                            actions: _Unwind_Action,
                                            exception_class: _Unwind_Exception_Class,
                                            ue_header: *_Unwind_Exception,
                                            context: *_Unwind_Context) -> _Unwind_Reason_Code {
    if (actions as c_int & _UA_SEARCH_PHASE as c_int) != 0 {
        // Always catch
        _URC_HANDLER_FOUND
    }
    else {
        // In cleanup phase delegate to __gcc_personality_v0 to install the landing pad context
        unsafe {
             __gcc_personality_v0(version, actions, exception_class, ue_header, context)
        }
    }
}

// can this be made a static var somehow?
fn rust_exception_class() -> _Unwind_Exception_Class {
    let bytes = bytes!("MOZ\0RUST");
    unsafe {
        let ptr: *_Unwind_Exception_Class = transmute(bytes.as_ptr());
        *ptr
    }
}
