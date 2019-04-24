// Ignore this test on Android, because it segfaults there.

// ignore-android
// ignore-windows
// ignore-cloudabi no execve
// ignore-emscripten no execve
// ignore-sgx no execve
// no-prefer-dynamic

#![feature(rustc_private)]

extern crate libc;

use libc::c_char;
use libc::execve;
use std::env;
use std::ffi::CString;
use std::os::unix::prelude::*;
use std::ptr;

fn main() {
    if env::args_os().count() == 2 {
        for (key, value) in env::vars_os() {
            panic!("found env value {:?} {:?}", key, value);
        }
        return;
    }

    let current_exe = CString::new(env::current_exe()
                                       .unwrap()
                                       .as_os_str()
                                       .as_bytes()).unwrap();
    let new_env_var = CString::new("FOOBAR").unwrap();
    let filename: *const c_char = current_exe.as_ptr();
    let argv: &[*const c_char] = &[filename, filename, ptr::null()];
    let envp: &[*const c_char] = &[new_env_var.as_ptr(), ptr::null()];
    unsafe {
        execve(filename, &argv[0], &envp[0]);
    }
    panic!("execve failed");
}
