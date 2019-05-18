// run-pass
#![allow(unused_imports)]
// ignore-cloudabi no processes
// ignore-emscripten no processes
// ignore-sgx no processes

use std::env;
use std::process::{self, Command, Stdio};

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() > 1 && args[1] == "child" {
        child();
    } else {
        parent();
    }
}

fn parent() {
    let args: Vec<String> = env::args().collect();
    let status = Command::new(&args[0]).arg("child").status().unwrap();
    assert_eq!(status.code(), Some(2));
}

fn child() -> i32 {
    process::exit(2);
}
