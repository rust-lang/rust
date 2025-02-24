//@ run-pass
//@ needs-subprocess
//@ compile-flags: -Zub-checks=no -Zmir-enable-passes=-CheckNull
//@ ignore-fuchsia must translate zircon signal to SIGSEGV/SIGBUS, FIXME (#58590)

#![feature(rustc_private)]

use std::env;
use std::ffi::c_char;
use std::process::{Command, ExitStatus};

#[link(name = "rust_test_helpers", kind = "static")]
extern "C" {
    fn rust_get_null_ptr() -> *mut c_char;
}

#[cfg(unix)]
fn check_status(status: ExitStatus) {
    extern crate libc;
    use std::os::unix::process::ExitStatusExt;

    assert!(status.signal() == Some(libc::SIGSEGV) || status.signal() == Some(libc::SIGBUS));
}

#[cfg(not(unix))]
fn check_status(status: ExitStatus) {
    assert!(!status.success());
}

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() > 1 && args[1] == "segfault" {
        unsafe {
            *rust_get_null_ptr() = 1;
        }; // trigger a segfault
    } else {
        let segfault = Command::new(&args[0]).arg("segfault").output().unwrap();
        let stderr = String::from_utf8_lossy(&segfault.stderr);
        let stdout = String::from_utf8_lossy(&segfault.stdout);
        println!("stdout: {}", stdout);
        println!("stderr: {}", stderr);
        println!("status: {}", segfault.status);
        check_status(segfault.status);
        assert!(!stderr.contains("has overflowed its stack"));
    }
}
