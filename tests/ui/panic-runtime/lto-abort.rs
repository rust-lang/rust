//@ ignore-backends: gcc
//@ run-pass
//@ compile-flags:-C lto -C panic=abort
//@ no-prefer-dynamic
//@ needs-subprocess

use std::process::Command;
use std::env;

struct Bomb;

impl Drop for Bomb {
    fn drop(&mut self) {
        std::process::exit(0);
    }
}

fn main() {
    let mut args = env::args_os();
    let _ = args.next().unwrap();

    if let Some(s) = args.next() {
        if &*s == "foo" {

            let _bomb = Bomb;

            panic!("try to catch me");
        }
    }
    let s = Command::new(env::args_os().next().unwrap()).arg("foo").status();
    assert!(s.unwrap().code() != Some(0));
}
