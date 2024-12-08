#![crate_type = "staticlib"]
#![feature(c_variadic)]

use std::ffi::{CStr, CString, VaList, c_char, c_double, c_int, c_long, c_longlong};

macro_rules! continue_if {
    ($cond:expr) => {
        if !($cond) {
            return 0xff;
        }
    };
}

unsafe fn compare_c_str(ptr: *const c_char, val: &str) -> bool {
    let cstr0 = CStr::from_ptr(ptr);
    match CString::new(val) {
        Ok(cstr1) => &*cstr1 == cstr0,
        Err(_) => false,
    }
}

#[no_mangle]
pub unsafe extern "C" fn check_list_0(mut ap: VaList) -> usize {
    continue_if!(ap.arg::<c_longlong>() == 1);
    continue_if!(ap.arg::<c_int>() == 2);
    continue_if!(ap.arg::<c_longlong>() == 3);
    0
}

#[no_mangle]
pub unsafe extern "C" fn check_list_1(mut ap: VaList) -> usize {
    continue_if!(ap.arg::<c_int>() == -1);
    continue_if!(ap.arg::<c_char>() == 'A' as c_char);
    continue_if!(ap.arg::<c_char>() == '4' as c_char);
    continue_if!(ap.arg::<c_char>() == ';' as c_char);
    continue_if!(ap.arg::<c_int>() == 0x32);
    continue_if!(ap.arg::<c_int>() == 0x10000001);
    continue_if!(compare_c_str(ap.arg::<*const c_char>(), "Valid!"));
    0
}

#[no_mangle]
pub unsafe extern "C" fn check_list_2(mut ap: VaList) -> usize {
    continue_if!(ap.arg::<c_double>().floor() == 3.14f64.floor());
    continue_if!(ap.arg::<c_long>() == 12);
    continue_if!(ap.arg::<c_char>() == 'a' as c_char);
    continue_if!(ap.arg::<c_double>().floor() == 6.18f64.floor());
    continue_if!(compare_c_str(ap.arg::<*const c_char>(), "Hello"));
    continue_if!(ap.arg::<c_int>() == 42);
    continue_if!(compare_c_str(ap.arg::<*const c_char>(), "World"));
    0
}

#[no_mangle]
pub unsafe extern "C" fn check_list_copy_0(mut ap: VaList) -> usize {
    continue_if!(ap.arg::<c_double>().floor() == 6.28f64.floor());
    continue_if!(ap.arg::<c_int>() == 16);
    continue_if!(ap.arg::<c_char>() == 'A' as c_char);
    continue_if!(compare_c_str(ap.arg::<*const c_char>(), "Skip Me!"));
    ap.with_copy(
        |mut ap| {
            if compare_c_str(ap.arg::<*const c_char>(), "Correct") { 0 } else { 0xff }
        },
    )
}

#[no_mangle]
pub unsafe extern "C" fn check_varargs_0(_: c_int, mut ap: ...) -> usize {
    continue_if!(ap.arg::<c_int>() == 42);
    continue_if!(compare_c_str(ap.arg::<*const c_char>(), "Hello, World!"));
    0
}

#[no_mangle]
pub unsafe extern "C" fn check_varargs_1(_: c_int, mut ap: ...) -> usize {
    continue_if!(ap.arg::<c_double>().floor() == 3.14f64.floor());
    continue_if!(ap.arg::<c_long>() == 12);
    continue_if!(ap.arg::<c_char>() == 'A' as c_char);
    continue_if!(ap.arg::<c_longlong>() == 1);
    0
}

#[no_mangle]
pub unsafe extern "C" fn check_varargs_2(_: c_int, _ap: ...) -> usize {
    0
}

#[no_mangle]
pub unsafe extern "C" fn check_varargs_3(_: c_int, mut ap: ...) -> usize {
    continue_if!(ap.arg::<c_int>() == 1);
    continue_if!(ap.arg::<c_int>() == 2);
    continue_if!(ap.arg::<c_int>() == 3);
    continue_if!(ap.arg::<c_int>() == 4);
    continue_if!(ap.arg::<c_int>() == 5);
    continue_if!(ap.arg::<c_int>() == 6);
    continue_if!(ap.arg::<c_int>() == 7);
    continue_if!(ap.arg::<c_int>() == 8);
    continue_if!(ap.arg::<c_int>() == 9);
    continue_if!(ap.arg::<c_int>() == 10);
    0
}

#[no_mangle]
pub unsafe extern "C" fn check_varargs_4(_: c_double, mut ap: ...) -> usize {
    continue_if!(ap.arg::<c_double>() == 1.0);
    continue_if!(ap.arg::<c_double>() == 2.0);
    continue_if!(ap.arg::<c_double>() == 3.0);
    continue_if!(ap.arg::<c_double>() == 4.0);
    continue_if!(ap.arg::<c_double>() == 5.0);
    continue_if!(ap.arg::<c_double>() == 6.0);
    continue_if!(ap.arg::<c_double>() == 7.0);
    continue_if!(ap.arg::<c_double>() == 8.0);
    continue_if!(ap.arg::<c_double>() == 9.0);
    continue_if!(ap.arg::<c_double>() == 10.0);
    0
}

#[no_mangle]
pub unsafe extern "C" fn check_varargs_5(_: c_int, mut ap: ...) -> usize {
    continue_if!(ap.arg::<c_double>() == 1.0);
    continue_if!(ap.arg::<c_int>() == 1);
    continue_if!(ap.arg::<c_double>() == 2.0);
    continue_if!(ap.arg::<c_int>() == 2);
    continue_if!(ap.arg::<c_double>() == 3.0);
    continue_if!(ap.arg::<c_int>() == 3);
    continue_if!(ap.arg::<c_double>() == 4.0);
    continue_if!(ap.arg::<c_int>() == 4);
    continue_if!(ap.arg::<c_int>() == 5);
    continue_if!(ap.arg::<c_double>() == 5.0);
    continue_if!(ap.arg::<c_int>() == 6);
    continue_if!(ap.arg::<c_double>() == 6.0);
    continue_if!(ap.arg::<c_int>() == 7);
    continue_if!(ap.arg::<c_double>() == 7.0);
    continue_if!(ap.arg::<c_int>() == 8);
    continue_if!(ap.arg::<c_double>() == 8.0);
    continue_if!(ap.arg::<c_int>() == 9);
    continue_if!(ap.arg::<c_double>() == 9.0);
    continue_if!(ap.arg::<c_int>() == 10);
    continue_if!(ap.arg::<c_double>() == 10.0);
    0
}
