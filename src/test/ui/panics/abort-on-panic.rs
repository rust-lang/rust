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

fn should_have_aborted() {
    io::stdout().write(b"This should never be printed.\n");
    let _ = io::stdout().flush();
}

fn test() {
    let _ = panic::catch_unwind(|| { panic_in_ffi(); });
    should_have_aborted();
}

fn testrust() {
    let _ = panic::catch_unwind(|| { panic_in_rust_abi(); });
    should_have_aborted();
}

fn main() {
    let tests: &[(_, fn())] = &[
        ("test", test),
        ("testrust", testrust),
    ];

    let args: Vec<String> = env::args().collect();
    if args.len() > 1 {
        // This is inside the self-executed command.
        for (a,f) in tests {
            if &args[1] == a { return f() }
        }
        panic!("bad test");
    }

    let execute_self_expecting_abort = |arg| {
        let mut p = Command::new(&args[0])
                            .stdout(Stdio::piped())
                            .stdin(Stdio::piped())
                            .arg(arg).spawn().unwrap();
        assert!(!p.wait().unwrap().success());
    };

    for (a,_f) in tests {
        execute_self_expecting_abort(a);
    }
}
