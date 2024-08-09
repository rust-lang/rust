#![crate_type = "lib"]
#![deny(improper_ctypes, improper_ctypes_definitions)]

use std::ffi::{CStr, CString};

extern "C" {
    fn take_cstr(s: CStr);
    //~^ ERROR `extern` block uses type `CStr`, which is not FFI-safe
    //~| HELP consider passing a `*const std::ffi::c_char` instead, and use `CStr::as_ptr()`
    fn take_cstr_ref(s: &CStr);
    //~^ ERROR `extern` block uses type `CStr`, which is not FFI-safe
    //~| HELP consider passing a `*const std::ffi::c_char` instead, and use `CStr::as_ptr()`
    fn take_cstring(s: CString);
    //~^ ERROR `extern` block uses type `CString`, which is not FFI-safe
    //~| HELP consider passing a `*const std::ffi::c_char` instead, and use `CStr::as_ptr()`
    fn take_cstring_ref(s: &CString);
    //~^ ERROR `extern` block uses type `CString`, which is not FFI-safe
    //~| HELP consider passing a `*const std::ffi::c_char` instead, and use `CStr::as_ptr()`

    fn no_special_help_for_mut_cstring(s: *mut CString);
    //~^ ERROR `extern` block uses type `CString`, which is not FFI-safe
    //~| HELP consider adding a `#[repr(C)]` or `#[repr(transparent)]` attribute to this struct

    fn no_special_help_for_mut_cstring_ref(s: &mut CString);
    //~^ ERROR `extern` block uses type `CString`, which is not FFI-safe
    //~| HELP consider adding a `#[repr(C)]` or `#[repr(transparent)]` attribute to this struct
}

extern "C" fn rust_take_cstr_ref(s: &CStr) {}
//~^ ERROR `extern` fn uses type `CStr`, which is not FFI-safe
//~| HELP consider passing a `*const std::ffi::c_char` instead, and use `CStr::as_ptr()`
extern "C" fn rust_take_cstring(s: CString) {}
//~^ ERROR `extern` fn uses type `CString`, which is not FFI-safe
//~| HELP consider passing a `*const std::ffi::c_char` instead, and use `CStr::as_ptr()`
extern "C" fn rust_no_special_help_for_mut_cstring(s: *mut CString) {}
extern "C" fn rust_no_special_help_for_mut_cstring_ref(s: &mut CString) {}
