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
use std::cell::Cell;

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

fn in_tls_destructor(f: impl FnOnce() + 'static) {
    struct RunOnDrop(Cell<Option<Box<dyn FnOnce() + 'static>>>);
    impl Drop for RunOnDrop {
        fn drop(&mut self) {
            self.0.take().unwrap()()
        }
    }

    thread_local! {
        static RUN: RunOnDrop = RunOnDrop(Cell::new(None));
    }

    RUN.with(|run| run.0.set(Some(Box::new(f))));
}

#[cfg(unix)]
fn check_status(status: std::process::ExitStatus)
{
    use std::os::unix::process::ExitStatusExt;

    assert!(!status.success());
    #[cfg(not(target_vendor = "apple"))]
    assert_eq!(status.signal(), Some(libc::SIGABRT));

    // Apple's libc has a bug where calling abort in a TLS destructor on a thread
    // other than the main thread results in a SIGTRAP instead of a SIGABRT.
    #[cfg(target_vendor = "apple")]
    assert!(matches!(status.signal(), Some(libc::SIGABRT | libc::SIGTRAP)));
}

#[cfg(not(unix))]
fn check_status(status: std::process::ExitStatus)
{
    assert!(!status.success());
}

fn main() {
    let args: Vec<String> = env::args().collect();
    match args.get(1).map(String::as_str) {
        Some("silent") => silent_recurse(),
        Some("loud") => loud_recurse(),
        Some("silent-thread") => thread::spawn(silent_recurse).join().unwrap(),
        Some("loud-thread") => thread::spawn(loud_recurse).join().unwrap(),
        Some("silent-tls") => in_tls_destructor(silent_recurse),
        Some("loud-tls") => in_tls_destructor(loud_recurse),
        Some("silent-thread-tls") => {
            thread::spawn(|| in_tls_destructor(silent_recurse)).join().unwrap();
        }
        Some("loud-thread-tls") => {
            thread::spawn(|| in_tls_destructor(loud_recurse)).join().unwrap();
        }
        _ => {
            let mut modes = vec![
                "silent-thread",
                "loud-thread",
                "silent-thread-tls",
                "loud-thread-tls",
            ];

            // On linux it looks like the main thread can sometimes grow its stack
            // basically without bounds, so we only test the child thread cases
            // there.
            if !cfg!(target_os = "linux") {
                modes.extend(["silent", "loud", "silent-tls", "loud-tls"]);
            }

            for mode in modes {
                println!("testing: {}", mode);

                let silent = Command::new(&args[0]).arg(mode).output().unwrap();

                let error = String::from_utf8_lossy(&silent.stderr);
                assert!(error.contains("has overflowed its stack"),
                        "missing overflow message: {}", error);

                check_status(silent.status);
            }
        }
    }
}
