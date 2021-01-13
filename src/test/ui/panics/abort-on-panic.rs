// run-pass

#![allow(unused_must_use)]
#![feature(unwind_attributes)]
// Since we mark some ABIs as "nounwind" to LLVM, we must make sure that
// we never unwind through them.

// ignore-emscripten no processes
// ignore-sgx no processes

use std::{env, panic};
use std::io::prelude::*;
use std::io;
use std::process::{Command, Stdio};

#[unwind(aborts)] // FIXME(#58794) should work even without the attribute
extern "C" fn panic_in_ffi() {
    panic!("Test");
}

#[unwind(aborts)]
extern "Rust" fn panic_in_rust_abi() {
    panic!("TestRust");
}

fn test() {
    let _ = panic::catch_unwind(|| { panic_in_ffi(); });
    // The process should have aborted by now.
    io::stdout().write(b"This should never be printed.\n");
    let _ = io::stdout().flush();
}

fn testrust() {
    let _ = panic::catch_unwind(|| { panic_in_rust_abi(); });
    // The process should have aborted by now.
    io::stdout().write(b"This should never be printed.\n");
    let _ = io::stdout().flush();
}

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() > 1 {
        // This is inside the self-executed command.
        match &*args[1] {
            "test" => return test(),
            "testrust" => return testrust(),
            _ => panic!("bad test"),
        }
    }

    // These end up calling the self-execution branches above.
    let mut p = Command::new(&args[0])
                        .stdout(Stdio::piped())
                        .stdin(Stdio::piped())
                        .arg("test").spawn().unwrap();
    assert!(!p.wait().unwrap().success());

    let mut p = Command::new(&args[0])
                        .stdout(Stdio::piped())
                        .stdin(Stdio::piped())
                        .arg("testrust").spawn().unwrap();
    assert!(!p.wait().unwrap().success());
}
