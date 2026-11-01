//@ run-pass
//@ edition: 2021

//@ ignore-android segfaults
//@ ignore-windows
//@ ignore-wasm32 no execve
//@ ignore-sgx no execve
//@ ignore-vxworks no execve
//@ ignore-fuchsia no 'execve'
//@ no-prefer-dynamic

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
            if key == "DYLD_ROOT_PATH" {
                continue;
            }
            panic!("found env value {:?} {:?}", key, value);
        }
        return;
    }

    let current_exe = CString::new(env::current_exe()
                                       .unwrap()
                                       .as_os_str()
                                       .as_bytes()).unwrap();
    let filename: *const c_char = current_exe.as_ptr();
    let argv: &[*const c_char] = &[filename, filename, ptr::null()];

    let root;
    let envp: &[*const c_char] = if cfg!(all(target_vendor = "apple", target_env = "sim")) {
        // Workaround: iOS/tvOS/watchOS/visionOS simulators need the root path
        // from the current process.
        root = format!("DYLD_ROOT_PATH={}\0", std::env::var("DYLD_ROOT_PATH").unwrap());
        &[c"FOOBAR".as_ptr(), root.as_ptr().cast(), ptr::null()]
    } else {
        // Try to set an environment variable without a value.
        &[c"FOOBAR".as_ptr(), ptr::null()]
    };

    unsafe {
        execve(filename, &argv[0], &envp[0]);
    }
    panic!("execve failed");
}
