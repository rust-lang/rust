//@ run-pass
//@ compile-flags:-C panic=abort
//@ no-prefer-dynamic
//@ needs-subprocess
//@ ignore-backends: gcc

use std::env;
use std::process::Command;

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

    let mut cmd = Command::new(env::args_os().next().unwrap());
    cmd.arg("foo");

    // ARMv6 hanges while printing the backtrace, see #41004
    if cfg!(target_arch = "arm") && cfg!(target_env = "gnu") {
        cmd.env("RUST_BACKTRACE", "0");
    }

    let s = cmd.status();
    assert!(s.unwrap().code() != Some(0));
}
