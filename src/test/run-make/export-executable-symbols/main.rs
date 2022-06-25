// edition:2018

#![feature(rustc_private)]

extern crate libc;
use std::ffi::*;
use std::os::unix::ffi::*;

fn main() {
    let path = std::env::var("TMPDIR").unwrap();
    let path = std::path::PathBuf::from(path).join("libfoo.so");

    let s = CString::new(path.as_os_str().as_bytes()).unwrap();
    let handle = unsafe { libc::dlopen(s.as_ptr(), libc::RTLD_LAZY | libc::RTLD_GLOBAL) };
    if handle.is_null() {
        let msg = unsafe { CStr::from_ptr(libc::dlerror() as *const _) };
        panic!("failed to dlopen lib {:?}", msg);
    }

    unsafe {
        libc::dlerror();
    }

    let raw_string = CString::new("call_exported_symbol").unwrap();
    let symbol = unsafe { libc::dlsym(handle as *mut libc::c_void, raw_string.as_ptr()) };
    if symbol.is_null() {
        let msg = unsafe { CStr::from_ptr(libc::dlerror() as *const _) };
        panic!("failed to load symbol {:?}", msg);
    }
    let func: extern "C" fn() -> i8 = unsafe { std::mem::transmute(symbol) };
    assert_eq!(func(), 42);
}

#[no_mangle]
pub fn exported_symbol() -> i8 {
    42
}
