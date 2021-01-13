// run-pass
#![allow(unused_variables)]

// compile-flags:-C lto -C panic=unwind
// no-prefer-dynamic
// ignore-emscripten no processes
// ignore-sgx no processes

use std::process::Command;
use std::env;

struct Bomb;

impl Drop for Bomb {
    fn drop(&mut self) {
        println!("hurray you ran me");
    }
}

fn main() {
    let mut args = env::args_os();
    let me = args.next().unwrap();

    if let Some(s) = args.next() {
        if &*s == "foo" {

            let _bomb = Bomb;

            panic!("try to catch me");
        }
    }
    let s = Command::new(env::args_os().next().unwrap()).arg("foo").output();
    let s = s.unwrap();
    assert!(!s.status.success());
    assert!(String::from_utf8_lossy(&s.stdout).contains("hurray you ran me"));
}
