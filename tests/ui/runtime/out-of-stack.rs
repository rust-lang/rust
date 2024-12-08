//@ run-pass

#![allow(unused_must_use)]
#![allow(unconditional_recursion)]
//@ ignore-android: FIXME (#20004)
//@ ignore-wasm32 no processes
//@ ignore-sgx no processes
//@ ignore-fuchsia must translate zircon signal to SIGABRT, FIXME (#58590)
//@ ignore-nto no stack overflow handler used (no alternate stack available)
//@ ignore-ios stack overflow handlers aren't enabled
//@ ignore-tvos stack overflow handlers aren't enabled
//@ ignore-watchos stack overflow handlers aren't enabled
//@ ignore-visionos stack overflow handlers aren't enabled

#![feature(rustc_private)]

#[cfg(unix)]
extern crate libc;

use std::env;
use std::hint::black_box;
use std::process::Command;
use std::thread;

fn silent_recurse() {
    let buf = [0u8; 1000];
    black_box(buf);
    silent_recurse();
}

fn loud_recurse() {
    println!("hello!");
    loud_recurse();
    black_box(()); // don't optimize this into a tail call. please.
}

#[cfg(unix)]
fn check_status(status: std::process::ExitStatus)
{
    use std::os::unix::process::ExitStatusExt;

    assert!(!status.success());
    assert_eq!(status.signal(), Some(libc::SIGABRT));
}

#[cfg(not(unix))]
fn check_status(status: std::process::ExitStatus)
{
    assert!(!status.success());
}


fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() > 1 && args[1] == "silent" {
        silent_recurse();
    } else if args.len() > 1 && args[1] == "loud" {
        loud_recurse();
    } else if args.len() > 1 && args[1] == "silent-thread" {
        thread::spawn(silent_recurse).join();
    } else if args.len() > 1 && args[1] == "loud-thread" {
        thread::spawn(loud_recurse).join();
    } else {
        let mut modes = vec![
            "silent-thread",
            "loud-thread",
        ];

        // On linux it looks like the main thread can sometimes grow its stack
        // basically without bounds, so we only test the child thread cases
        // there.
        if !cfg!(target_os = "linux") {
            modes.push("silent");
            modes.push("loud");
        }
        for mode in modes {
            println!("testing: {}", mode);

            let silent = Command::new(&args[0]).arg(mode).output().unwrap();

            check_status(silent.status);

            let error = String::from_utf8_lossy(&silent.stderr);
            assert!(error.contains("has overflowed its stack"),
                    "missing overflow message: {}", error);
        }
    }
}
