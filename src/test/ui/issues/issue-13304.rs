// run-pass
#![allow(unused_mut)]
// ignore-emscripten no processes
// ignore-sgx no processes

use std::env;
use std::io::prelude::*;
use std::io;
use std::process::{Command, Stdio};
use std::str;

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
    let mut p = Command::new(&args[0]).arg("child")
                        .stdout(Stdio::piped())
                        .stdin(Stdio::piped())
                        .spawn().unwrap();
    p.stdin.as_mut().unwrap().write_all(b"test1\ntest2\ntest3").unwrap();
    let out = p.wait_with_output().unwrap();
    assert!(out.status.success());
    let s = str::from_utf8(&out.stdout).unwrap();
    assert_eq!(s, "test1\ntest2\ntest3\n");
}

fn child() {
    let mut stdin = io::stdin();
    for line in stdin.lock().lines() {
        println!("{}", line.unwrap());
    }
}
