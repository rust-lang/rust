// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![allow(private_no_mangle_fns)]

use prelude::v1::*;

use any::Any;
use sys_common::libunwind as uw;

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
        unsafe {
            let _: Box<Exception> = Box::from_raw(exception as *mut Exception);
        }
    }
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
// See also: rustc_trans::trans::intrinsic::trans_gnu_try

#[cfg(all(not(target_arch = "arm"),
          not(all(windows, target_arch = "x86_64")),
          not(test)))]
pub mod eabi {
    use sys_common::libunwind as uw;
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
    use sys_common::libunwind as uw;
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
    use sys_common::libunwind as uw;
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

// See docs in the `unwind` module.
#[cfg(all(target_os="windows", target_arch = "x86", target_env="gnu", not(test)))]
#[lang = "eh_unwind_resume"]
#[unwind]
unsafe extern fn rust_eh_unwind_resume(panic_ctx: *mut u8) -> ! {
    uw::_Unwind_Resume(panic_ctx as *mut uw::_Unwind_Exception);
}

#[cfg(all(target_os="windows", target_arch = "x86", target_env="gnu"))]
pub mod eh_frame_registry {
    // The implementation of stack unwinding is (for now) deferred to libgcc_eh, however Rust
    // crates use these Rust-specific entry points to avoid potential clashes with GCC runtime.
    // See also: rtbegin.rs, `unwind` module.

    #[link(name = "gcc_eh")]
    extern {
        fn __register_frame_info(eh_frame_begin: *const u8, object: *mut u8);
        fn __deregister_frame_info(eh_frame_begin: *const u8, object: *mut u8);
    }
    #[cfg(not(test))]
    #[no_mangle]
    #[unstable(feature = "libstd_sys_internals", issue = "0")]
    pub unsafe extern fn rust_eh_register_frames(eh_frame_begin: *const u8,
                                                 object: *mut u8) {
        __register_frame_info(eh_frame_begin, object);
    }
    #[cfg(not(test))]
    #[no_mangle]
    #[unstable(feature = "libstd_sys_internals", issue = "0")]
    pub  unsafe extern fn rust_eh_unregister_frames(eh_frame_begin: *const u8,
                                                   object: *mut u8) {
        __deregister_frame_info(eh_frame_begin, object);
    }
}
