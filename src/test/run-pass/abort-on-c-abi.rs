#![allow(unused_must_use)]
// Since we mark some ABIs as "nounwind" to LLVM, we must make sure that
// we never unwind through them.

// ignore-cloudabi no env and process
// ignore-emscripten no processes
// ignore-sgx no processes

use std::{env, panic};
use std::io::prelude::*;
use std::io;
use std::process::{Command, Stdio};

extern "C" fn panic_in_ffi() {
    panic!("Test");
}

fn test() {
    let _ = panic::catch_unwind(|| { panic_in_ffi(); });
    // The process should have aborted by now.
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
}
