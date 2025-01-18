#![crate_type = "lib"]
#![deny(improper_ctypes, improper_ctypes_definitions)]

use std::ffi::{CStr, CString};

extern "C" {
    fn take_cstr(s: CStr);
    //~^ ERROR `extern` block uses type `CStr`, which is not FFI-safe
    //~| HELP consider passing a `*const std::ffi::c_char` or `*mut std::ffi::c_char` instead
    fn take_cstr_ref(s: &CStr);
    //~^ ERROR `extern` block uses type `&CStr`, which is not FFI-safe
    //~| HELP consider passing a `*const std::ffi::c_char` instead, and use `CStr::as_ptr()`
    fn take_cstring(s: CString);
    //~^ ERROR `extern` block uses type `CString`, which is not FFI-safe
    //~| HELP consider passing a `*const std::ffi::c_char` or `*mut std::ffi::c_char` instead
    fn take_cstring_ref(s: &CString);
    //~^ ERROR `extern` block uses type `CString`, which is not FFI-safe
    //~| HELP consider passing a `*const std::ffi::c_char` instead, and use `CStr::as_ptr()`

    fn take_cstring_ptr_mut(s: *mut CString);
    //~^ ERROR `extern` block uses type `CString`, which is not FFI-safe
    //~| HELP consider passing a `*mut std::ffi::c_char` instead, and use `CString::into_raw()` then `CString::from_raw()` or a dedicated buffer

    fn take_cstring_ref_mut(s: &mut CString);
    //~^ ERROR `extern` block uses type `CString`, which is not FFI-safe
    //~| HELP consider passing a `*mut std::ffi::c_char` instead, and use `CString::into_raw()` then `CString::from_raw()` or a dedicated buffer
}

extern "C" fn rust_take_cstr_ref(s: &CStr) {}
//~^ ERROR `extern` fn uses type `&CStr`, which is not FFI-safe
//~| HELP consider passing a `*const std::ffi::c_char` instead, and use `CStr::as_ptr()`
extern "C" fn rust_take_cstring(s: CString) {}
//~^ ERROR `extern` fn uses type `CString`, which is not FFI-safe
//~| HELP consider passing a `*const std::ffi::c_char` or `*mut std::ffi::c_char` instead
extern "C" fn rust_take_cstring_ptr_mut(s: *mut CString) {}
//~^ ERROR `extern` fn uses type `CString`, which is not FFI-safe
//~| HELP consider passing a `*mut std::ffi::c_char` instead, and use `CString::into_raw()` then `CString::from_raw()` or a dedicated buffer
extern "C" fn rust_take_cstring_ref_mut(s: &mut CString) {}
//~^ ERROR `extern` fn uses type `CString`, which is not FFI-safe
//~| HELP consider passing a `*mut std::ffi::c_char` instead, and use `CString::into_raw()` then `CString::from_raw()` or a dedicated buffer
