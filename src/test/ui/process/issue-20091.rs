// run-pass
#![allow(stable_features)]

// ignore-emscripten no processes
// ignore-sgx no processes

#![feature(os)]

#[cfg(unix)]
fn main() {
    use std::process::Command;
    use std::env;
    use std::os::unix::prelude::*;
    use std::ffi::OsStr;

    if env::args().len() == 1 {
        assert!(Command::new(&env::current_exe().unwrap())
                        .arg(<OsStr as OsStrExt>::from_bytes(b"\xff"))
                        .status().unwrap().success())
    }
}

#[cfg(windows)]
fn main() {}
