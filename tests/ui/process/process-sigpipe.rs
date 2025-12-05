//@ run-pass
#![allow(unused_imports)]
#![allow(deprecated)]

//@ ignore-android since the dynamic linker sets a SIGPIPE handler (to do
// a crash report) so inheritance is moot on the entire platform

// libstd ignores SIGPIPE, and other libraries may set signal masks.
// Make sure that these behaviors don't get inherited to children
// spawned via std::process, since they're needed for traditional UNIX
// filter behavior.
// This test checks that `yes` or `while echo y ; do : ; done | head`
// terminates (instead of running forever), and that it does not print an
// error message about a broken pipe.

//@ ignore-vxworks no 'sh'
//@ ignore-fuchsia no 'sh'
//@ ignore-ios no 'sh'
//@ ignore-tvos no 'sh'
//@ ignore-watchos no 'sh'
//@ ignore-visionos no 'sh'
//@ needs-threads
//@ only-unix SIGPIPE is a unix feature

use std::process;
use std::thread;

fn main() {
    // Just in case `yes` or `while-echo` doesn't check for EPIPE...
    thread::spawn(|| {
        thread::sleep_ms(5000);
        process::exit(1);
    });
    // QNX Neutrino does not have `yes`. Therefore, use `while-echo` for `nto`
    // and `yes` for other platforms.
    let command = if cfg!(target_os = "nto") {
        "while echo y ; do : ; done | head"
    } else {
        "yes | head"
    };
    let output = process::Command::new("sh")
        .arg("-c")
        .arg(command)
        .output()
        .unwrap();
    assert!(output.status.success());
    assert!(output.stderr.len() == 0);
}
