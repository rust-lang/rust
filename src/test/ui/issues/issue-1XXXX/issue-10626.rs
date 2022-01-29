// run-pass
// ignore-emscripten no processes
// ignore-sgx no processes

// Make sure that if a process doesn't have its stdio/stderr descriptors set up
// that we don't die in a large ball of fire

use std::env;
use std::process::{Command, Stdio};

pub fn main () {
    let args: Vec<String> = env::args().collect();
    if args.len() > 1 && args[1] == "child" {
        for _ in 0..1000 {
            println!("hello?");
        }
        for _ in 0..1000 {
            println!("hello?");
        }
        return;
    }

    let mut p = Command::new(&args[0]);
    p.arg("child").stdout(Stdio::null()).stderr(Stdio::null());
    println!("{:?}", p.spawn().unwrap().wait());
}
