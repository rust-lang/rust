// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use prelude::v1::*;

use any::Any;
use libc::c_void;
use rt::libunwind as uw;

struct Exception {
    uwe: uw::_Unwind_Exception,
    cause: Option<Box<Any + Send + 'static>>,
}

pub unsafe fn panic(data: Box<Any + Send + 'static>) -> ! {
    let exception: Box<_> = box Exception {
        uwe: uw::_Unwind_Exception {
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
        rtdebug!("exception_cleanup()");
        unsafe {
            let _: Box<Exception> = Box::from_raw(exception as *mut Exception);
        }
    }
}

pub unsafe fn cleanup(ptr: *mut c_void) -> Box<Any + Send + 'static> {
    let my_ep = ptr as *mut Exception;
    rtdebug!("caught {}", (*my_ep).uwe.exception_class);
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

// We could implement our personality routine in pure Rust, however exception
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
// Note, however, that for implementation simplicity, rust_eh_personality_catch
// lacks code to install a landing pad, so in order to obtain exception object
// pointer (which it needs to return upstream), rust_try() employs another trick:
// it calls into the nested rust_try_inner(), whose landing pad does not resume
// unwinds.  Instead, it extracts the exception pointer and performs a "normal"
// return.
//
// See also: rt/rust_try.ll

#[cfg(all(not(target_arch = "arm"),
          not(all(windows, target_arch = "x86_64")),
          not(test)))]
pub mod eabi {
    use rt::libunwind as uw;
    use libc::c_int;

    extern "C" {
        fn __gcc_personality_v0(version: c_int,
                                actions: uw::_Unwind_Action,
                                exception_class: uw::_Unwind_Exception_Class,
                                ue_header: *mut uw::_Unwind_Exception,
                                context: *mut uw::_Unwind_Context)
            -> uw::_Unwind_Reason_Code;
    }

    #[lang="eh_personality"]
    #[no_mangle] // referenced from rust_try.ll
    #[allow(private_no_mangle_fns)]
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

    #[no_mangle] // referenced from rust_try.ll
    pub extern "C" fn rust_eh_personality_catch(
        _version: c_int,
        actions: uw::_Unwind_Action,
        _exception_class: uw::_Unwind_Exception_Class,
        _ue_header: *mut uw::_Unwind_Exception,
        _context: *mut uw::_Unwind_Context
    ) -> uw::_Unwind_Reason_Code
    {

        if (actions as c_int & uw::_UA_SEARCH_PHASE as c_int) != 0 { // search phase
            uw::_URC_HANDLER_FOUND // catch!
        }
        else { // cleanup phase
            uw::_URC_INSTALL_CONTEXT
        }
    }
}

// iOS on armv7 is using SjLj exceptions and therefore requires to use
// a specialized personality routine: __gcc_personality_sj0

#[cfg(all(target_os = "ios", target_arch = "arm", not(test)))]
pub mod eabi {
    use rt::libunwind as uw;
    use libc::c_int;

    extern "C" {
        fn __gcc_personality_sj0(version: c_int,
                                actions: uw::_Unwind_Action,
                                exception_class: uw::_Unwind_Exception_Class,
                                ue_header: *mut uw::_Unwind_Exception,
                                context: *mut uw::_Unwind_Context)
            -> uw::_Unwind_Reason_Code;
    }

    #[lang="eh_personality"]
    #[no_mangle] // referenced from rust_try.ll
    pub extern "C" fn rust_eh_personality(
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

    #[no_mangle] // referenced from rust_try.ll
    pub extern "C" fn rust_eh_personality_catch(
        _version: c_int,
        actions: uw::_Unwind_Action,
        _exception_class: uw::_Unwind_Exception_Class,
        _ue_header: *mut uw::_Unwind_Exception,
        _context: *mut uw::_Unwind_Context
    ) -> uw::_Unwind_Reason_Code
    {
        if (actions as c_int & uw::_UA_SEARCH_PHASE as c_int) != 0 { // search phase
            uw::_URC_HANDLER_FOUND // catch!
        }
        else { // cleanup phase
            unsafe {
                __gcc_personality_sj0(_version, actions, _exception_class, _ue_header,
                                      _context)
            }
        }
    }
}


// ARM EHABI uses a slightly different personality routine signature,
// but otherwise works the same.
#[cfg(all(target_arch = "arm", not(target_os = "ios"), not(test)))]
pub mod eabi {
    use rt::libunwind as uw;
    use libc::c_int;

    extern "C" {
        fn __gcc_personality_v0(state: uw::_Unwind_State,
                                ue_header: *mut uw::_Unwind_Exception,
                                context: *mut uw::_Unwind_Context)
            -> uw::_Unwind_Reason_Code;
    }

    #[lang="eh_personality"]
    #[no_mangle] // referenced from rust_try.ll
    #[allow(private_no_mangle_fns)]
    extern "C" fn rust_eh_personality(
        state: uw::_Unwind_State,
        ue_header: *mut uw::_Unwind_Exception,
        context: *mut uw::_Unwind_Context
    ) -> uw::_Unwind_Reason_Code
    {
        unsafe {
            __gcc_personality_v0(state, ue_header, context)
        }
    }

    #[no_mangle] // referenced from rust_try.ll
    pub extern "C" fn rust_eh_personality_catch(
        state: uw::_Unwind_State,
        _ue_header: *mut uw::_Unwind_Exception,
        _context: *mut uw::_Unwind_Context
    ) -> uw::_Unwind_Reason_Code
    {
        if (state as c_int & uw::_US_ACTION_MASK as c_int)
                           == uw::_US_VIRTUAL_UNWIND_FRAME as c_int { // search phase
            uw::_URC_HANDLER_FOUND // catch!
        }
        else { // cleanup phase
            uw::_URC_INSTALL_CONTEXT
        }
    }
}

// Win64 SEH (see http://msdn.microsoft.com/en-us/library/1eyas8tf.aspx)
//
// This looks a bit convoluted because rather than implementing a native SEH
// handler, GCC reuses the same personality routine as for the other
// architectures by wrapping it with an "API translator" layer
// (_GCC_specific_handler).

#[cfg(all(windows, target_arch = "x86_64", not(test)))]
#[doc(hidden)]
#[allow(non_camel_case_types, non_snake_case)]
pub mod eabi {
    pub use self::EXCEPTION_DISPOSITION::*;
    use rt::libunwind as uw;
    use libc::{c_void, c_int};

    #[repr(C)]
    pub struct EXCEPTION_RECORD;
    #[repr(C)]
    pub struct CONTEXT;
    #[repr(C)]
    pub struct DISPATCHER_CONTEXT;

    #[repr(C)]
    #[derive(Copy, Clone)]
    pub enum EXCEPTION_DISPOSITION {
        ExceptionContinueExecution,
        ExceptionContinueSearch,
        ExceptionNestedException,
        ExceptionCollidedUnwind
    }

    type _Unwind_Personality_Fn =
        extern "C" fn(
            version: c_int,
            actions: uw::_Unwind_Action,
            exception_class: uw::_Unwind_Exception_Class,
            ue_header: *mut uw::_Unwind_Exception,
            context: *mut uw::_Unwind_Context
        ) -> uw::_Unwind_Reason_Code;

    extern "C" {
        fn __gcc_personality_seh0(
            exceptionRecord: *mut EXCEPTION_RECORD,
            establisherFrame: *mut c_void,
            contextRecord: *mut CONTEXT,
            dispatcherContext: *mut DISPATCHER_CONTEXT
        ) -> EXCEPTION_DISPOSITION;

        fn _GCC_specific_handler(
            exceptionRecord: *mut EXCEPTION_RECORD,
            establisherFrame: *mut c_void,
            contextRecord: *mut CONTEXT,
            dispatcherContext: *mut DISPATCHER_CONTEXT,
            personality: _Unwind_Personality_Fn
        ) -> EXCEPTION_DISPOSITION;
    }

    #[lang="eh_personality"]
    #[no_mangle] // referenced from rust_try.ll
    #[allow(private_no_mangle_fns)]
    extern "C" fn rust_eh_personality(
        exceptionRecord: *mut EXCEPTION_RECORD,
        establisherFrame: *mut c_void,
        contextRecord: *mut CONTEXT,
        dispatcherContext: *mut DISPATCHER_CONTEXT
    ) -> EXCEPTION_DISPOSITION
    {
        unsafe {
            __gcc_personality_seh0(exceptionRecord, establisherFrame,
                                   contextRecord, dispatcherContext)
        }
    }

    #[no_mangle] // referenced from rust_try.ll
    pub extern "C" fn rust_eh_personality_catch(
        exceptionRecord: *mut EXCEPTION_RECORD,
        establisherFrame: *mut c_void,
        contextRecord: *mut CONTEXT,
        dispatcherContext: *mut DISPATCHER_CONTEXT
    ) -> EXCEPTION_DISPOSITION
    {
        extern "C" fn inner(
                _version: c_int,
                actions: uw::_Unwind_Action,
                _exception_class: uw::_Unwind_Exception_Class,
                _ue_header: *mut uw::_Unwind_Exception,
                _context: *mut uw::_Unwind_Context
            ) -> uw::_Unwind_Reason_Code
        {
            if (actions as c_int & uw::_UA_SEARCH_PHASE as c_int) != 0 { // search phase
                uw::_URC_HANDLER_FOUND // catch!
            }
            else { // cleanup phase
                uw::_URC_INSTALL_CONTEXT
            }
        }

        unsafe {
            _GCC_specific_handler(exceptionRecord, establisherFrame,
                                  contextRecord, dispatcherContext,
                                  inner)
        }
    }
}

