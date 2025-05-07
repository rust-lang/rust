//@revisions: edition2018 edition2021
//@[edition2018] edition:2018
//@[edition2021] edition:2021
//@[edition2018] check-pass
#![warn(clippy::manual_c_str_literals)]
#![allow(clippy::no_effect)]

use std::ffi::CStr;

macro_rules! cstr {
    ($s:literal) => {
        CStr::from_bytes_with_nul(concat!($s, "\0").as_bytes()).unwrap()
    };
}

macro_rules! macro_returns_c_str {
    () => {
        CStr::from_bytes_with_nul(b"foo\0").unwrap();
    };
}

macro_rules! macro_returns_byte_string {
    () => {
        b"foo\0"
    };
}

#[clippy::msrv = "1.76.0"]
fn pre_stabilization() {
    CStr::from_bytes_with_nul(b"foo\0");
}

#[clippy::msrv = "1.77.0"]
fn post_stabilization() {
    CStr::from_bytes_with_nul(b"foo\0");
    //~[edition2021]^ manual_c_str_literals
}

fn main() {
    CStr::from_bytes_with_nul(b"foo\0");
    //~[edition2021]^ manual_c_str_literals
    CStr::from_bytes_with_nul(b"foo\x00");
    //~[edition2021]^ manual_c_str_literals
    CStr::from_bytes_with_nul(b"foo\0").unwrap();
    //~[edition2021]^ manual_c_str_literals
    CStr::from_bytes_with_nul(b"foo\\0sdsd\0").unwrap();
    //~[edition2021]^ manual_c_str_literals
    CStr::from_bytes_with_nul(br"foo\\0sdsd\0").unwrap();
    CStr::from_bytes_with_nul(br"foo\x00").unwrap();
    CStr::from_bytes_with_nul(br##"foo#a\0"##).unwrap();

    unsafe { CStr::from_ptr(b"foo\0".as_ptr().cast()) };
    //~[edition2021]^ manual_c_str_literals
    unsafe { CStr::from_ptr(b"foo\0".as_ptr() as *const _) };
    //~[edition2021]^ manual_c_str_literals
    let _: *const _ = b"foo\0".as_ptr();
    //~[edition2021]^ manual_c_str_literals
    let _: *const _ = "foo\0".as_ptr();
    //~[edition2021]^ manual_c_str_literals
    let _: *const _ = "foo".as_ptr(); // not a C-string
    let _: *const _ = "".as_ptr();
    let _: *const _ = b"foo\0".as_ptr().cast::<i8>();
    //~[edition2021]^ manual_c_str_literals
    let _ = "电脑".as_ptr();
    let _ = "电脑\\".as_ptr();
    let _ = "电脑\\\0".as_ptr();
    //~[edition2021]^ manual_c_str_literals
    let _ = "电脑\0".as_ptr();
    //~[edition2021]^ manual_c_str_literals
    let _ = "电脑\x00".as_ptr();
    //~[edition2021]^ manual_c_str_literals

    // Macro cases, don't lint:
    cstr!("foo");
    macro_returns_c_str!();
    CStr::from_bytes_with_nul(macro_returns_byte_string!()).unwrap();
}
