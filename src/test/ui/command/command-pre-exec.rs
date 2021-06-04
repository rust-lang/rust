// run-pass
// revisions: mir thir
// [thir]compile-flags: -Zthir-unsafeck

#![allow(stable_features)]
// ignore-windows - this is a unix-specific test
// ignore-emscripten no processes
// ignore-sgx no processes
#![feature(process_exec, rustc_private)]

extern crate libc;

use std::env;
use std::io::Error;
use std::os::unix::process::CommandExt;
use std::process::Command;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

fn main() {
    if let Some(arg) = env::args().nth(1) {
        match &arg[..] {
            "test1" => println!("hello2"),
            "test2" => assert_eq!(env::var("FOO").unwrap(), "BAR"),
            "test3" => assert_eq!(env::current_dir().unwrap().to_str().unwrap(), "/"),
            "empty" => {}
            _ => panic!("unknown argument: {}", arg),
        }
        return;
    }

    let me = env::current_exe().unwrap();

    let output = unsafe {
        Command::new(&me)
            .arg("test1")
            .pre_exec(|| {
                println!("hello");
                Ok(())
            })
            .output()
            .unwrap()
    };
    assert!(output.status.success());
    assert!(output.stderr.is_empty());
    assert_eq!(output.stdout, b"hello\nhello2\n");

    let output = unsafe {
        Command::new(&me)
            .arg("test3")
            .pre_exec(|| {
                env::set_current_dir("/").unwrap();
                Ok(())
            })
            .output()
            .unwrap()
    };
    assert!(output.status.success());
    assert!(output.stderr.is_empty());
    assert!(output.stdout.is_empty());

    let output = unsafe {
        Command::new(&me)
            .arg("bad")
            .pre_exec(|| Err(Error::from_raw_os_error(102)))
            .output()
            .unwrap_err()
    };
    assert_eq!(output.raw_os_error(), Some(102));

    let pid = unsafe { libc::getpid() };
    assert!(pid >= 0);
    let output = unsafe {
        Command::new(&me)
            .arg("empty")
            .pre_exec(move || {
                let child = libc::getpid();
                assert!(child >= 0);
                assert!(pid != child);
                Ok(())
            })
            .output()
            .unwrap()
    };
    assert!(output.status.success());
    assert!(output.stderr.is_empty());
    assert!(output.stdout.is_empty());

    let mem = Arc::new(AtomicUsize::new(0));
    let mem2 = mem.clone();
    let output = unsafe {
        Command::new(&me)
            .arg("empty")
            .pre_exec(move || {
                assert_eq!(mem2.fetch_add(1, Ordering::SeqCst), 0);
                Ok(())
            })
            .output()
            .unwrap()
    };
    assert!(output.status.success());
    assert!(output.stderr.is_empty());
    assert!(output.stdout.is_empty());
    assert_eq!(mem.load(Ordering::SeqCst), 0);
}
