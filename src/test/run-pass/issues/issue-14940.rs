// run-pass
// ignore-cloudabi no processes
// ignore-emscripten no processes
// ignore-sgx no processes

use std::env;
use std::process::Command;
use std::io::{self, Write};

fn main() {
    let mut args = env::args();
    if args.len() > 1 {
        let mut out = io::stdout();
        out.write(&['a' as u8; 128 * 1024]).unwrap();
    } else {
        let out = Command::new(&args.next().unwrap()).arg("child").output();
        let out = out.unwrap();
        assert!(out.status.success());
    }
}
