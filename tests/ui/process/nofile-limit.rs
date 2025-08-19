// Check that statically linked binary executes successfully
// with RLIMIT_NOFILE resource lowered to zero. Regression
// test for issue #96621.
//
//@ run-pass
//@ dont-check-compiler-stderr
//@ only-linux
//@ no-prefer-dynamic
//@ compile-flags: -Ctarget-feature=+crt-static -Crpath=no -Crelocation-model=static
//@ ignore-backends: gcc

#![feature(exit_status_error)]
#![feature(rustc_private)]
extern crate libc;

use std::os::unix::process::CommandExt;
use std::process::Command;

fn main() {
    let mut args = std::env::args();
    let this = args.next().unwrap();
    match args.next().as_deref() {
        None => {
            let mut cmd = Command::new(this);
            cmd.arg("Ok!");
            unsafe {
                cmd.pre_exec(|| {
                    let rlim = libc::rlimit {
                        rlim_cur: 0,
                        rlim_max: 0,
                    };
                    if libc::setrlimit(libc::RLIMIT_NOFILE, &rlim) == -1 {
                        Err(std::io::Error::last_os_error())
                    } else {
                        Ok(())
                    }
                })
            };
            let output = cmd.output().unwrap();
            println!("{:?}", output);
            output.status.exit_ok().unwrap();
            assert!(output.stdout.starts_with(b"Ok!"));
        }
        Some(word) => {
            println!("{}", word);
        }
    }
}
