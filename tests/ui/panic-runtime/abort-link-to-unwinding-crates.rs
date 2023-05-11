// run-pass
#![allow(unused_variables)]
// compile-flags:-C panic=abort
// aux-build:exit-success-if-unwind.rs
// no-prefer-dynamic
// ignore-emscripten no processes
// ignore-sgx no processes
// ignore-macos

extern crate exit_success_if_unwind;

use std::process::Command;
use std::env;

fn main() {
    let mut args = env::args_os();
    let me = args.next().unwrap();

    if let Some(s) = args.next() {
        if &*s == "foo" {
            exit_success_if_unwind::bar(do_panic);
        }
    }

    let mut cmd = Command::new(env::args_os().next().unwrap());
    cmd.arg("foo");


    // ARMv6 hanges while printing the backtrace, see #41004
    if cfg!(target_arch = "arm") && cfg!(target_env = "gnu") {
        cmd.env("RUST_BACKTRACE", "0");
    }

    let s = cmd.status();
    assert!(s.unwrap().code() != Some(0));
}

fn do_panic() {
    panic!("try to catch me");
}
