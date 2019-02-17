//! Unwinding for *emscripten* target.
//!
//! Whereas Rust's usual unwinding implementation for Unix platforms
//! calls into the libunwind APIs directly, on Emscripten we instead
//! call into the C++ unwinding APIs. This is just an expedience since
//! Emscripten's runtime always implements those APIs and does not
//! implement libunwind.

#![allow(private_no_mangle_fns)]

use core::any::Any;
use core::ptr;
use core::mem;
use alloc::boxed::Box;
use libc::{self, c_int};
use unwind as uw;

pub fn payload() -> *mut u8 {
    ptr::null_mut()
}

pub unsafe fn cleanup(ptr: *mut u8) -> Box<dyn Any + Send> {
    assert!(!ptr.is_null());
    let ex = ptr::read(ptr as *mut _);
    __cxa_free_exception(ptr as *mut _);
    ex
}

pub unsafe fn panic(data: Box<dyn Any + Send>) -> u32 {
    let sz = mem::size_of_val(&data);
    let exception = __cxa_allocate_exception(sz);
    if exception == ptr::null_mut() {
        return uw::_URC_FATAL_PHASE1_ERROR as u32;
    }
    let exception = exception as *mut Box<dyn Any + Send>;
    ptr::write(exception, data);
    __cxa_throw(exception as *mut _, ptr::null_mut(), ptr::null_mut());

    unreachable!()
}

#[lang = "eh_personality"]
#[no_mangle]
unsafe extern "C" fn rust_eh_personality(version: c_int,
                                         actions: uw::_Unwind_Action,
                                         exception_class: uw::_Unwind_Exception_Class,
                                         exception_object: *mut uw::_Unwind_Exception,
                                         context: *mut uw::_Unwind_Context)
                                         -> uw::_Unwind_Reason_Code {
    __gxx_personality_v0(version, actions, exception_class, exception_object, context)
}

extern "C" {
    fn __cxa_allocate_exception(thrown_size: libc::size_t) -> *mut libc::c_void;
    fn __cxa_free_exception(thrown_exception: *mut libc::c_void);
    fn __cxa_throw(thrown_exception: *mut libc::c_void,
                   tinfo: *mut libc::c_void,
                   dest: *mut libc::c_void);
    fn __gxx_personality_v0(version: c_int,
                            actions: uw::_Unwind_Action,
                            exception_class: uw::_Unwind_Exception_Class,
                            exception_object: *mut uw::_Unwind_Exception,
                            context: *mut uw::_Unwind_Context)
                            -> uw::_Unwind_Reason_Code;
}
