// run-pass

#![allow(unused_must_use)]
#![feature(unwind_attributes)]
// Since we mark some ABIs as "nounwind" to LLVM, we must make sure that
// we never unwind through them.

// ignore-cloudabi no env and process
// ignore-emscripten no processes
// ignore-sgx no processes

use std::{env, panic};
use std::io::prelude::*;
use std::io;
use std::process::{Command, Stdio};

unsafe extern "C" fn unsafe_panic_in_ffi() {
    panic!("Test");
}

extern "C" fn panic_in_ffi() {
    panic!("Test");
}

fn test() {
    // A safe extern "C" function that panics should abort the process:
    let _ = panic::catch_unwind(|| panic_in_ffi() );

    // If the process did not abort, the panic escaped FFI:
    io::stdout().write(b"This should never be printed.\n");
    let _ = io::stdout().flush();
}

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() > 1 && args[1] == "test" {
        return test();
    }

    let mut p = Command::new(&args[0])
                        .stdout(Stdio::piped())
                        .stdin(Stdio::piped())
                        .arg("test").spawn().unwrap();
    assert!(!p.wait().unwrap().success());

    // An unsafe extern "C" function that panics should let the panic escape:
    assert!(panic::catch_unwind(|| unsafe { unsafe_panic_in_ffi() }).is_err());
}
