//@ run-pass
//@ needs-subprocess

use std::env;
use std::process::{self, Command};

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
