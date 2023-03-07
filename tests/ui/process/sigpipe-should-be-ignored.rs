// run-pass

#![allow(unused_must_use)]
// Be sure that when a SIGPIPE would have been received that the entire process
// doesn't die in a ball of fire, but rather it's gracefully handled.

// ignore-emscripten no processes
// ignore-sgx no processes

use std::env;
use std::io::prelude::*;
use std::io;
use std::process::{Command, Stdio};

fn test() {
    let _ = io::stdin().read_line(&mut String::new());
    io::stdout().write(&[1]);
    assert!(io::stdout().flush().is_err());
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
    drop(p.stdout.take());
    assert!(p.wait().unwrap().success());
}
