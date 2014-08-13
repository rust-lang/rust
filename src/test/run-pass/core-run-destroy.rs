// Copyright 2012-2013-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// ignore-pretty
// compile-flags:--test

// NB: These tests kill child processes. Valgrind sees these children as leaking
// memory, which makes for some *confusing* logs. That's why these are here
// instead of in std.

#![feature(macro_rules)]
#![reexport_test_harness_main = "test_main"]

extern crate libc;

extern crate native;
extern crate green;
extern crate rustuv;

use std::io::{Process, Command};
use std::time::Duration;

macro_rules! succeed( ($e:expr) => (
    match $e { Ok(..) => {}, Err(e) => fail!("failure: {}", e) }
) )

macro_rules! iotest (
    { fn $name:ident() $b:block $($a:attr)* } => (
        mod $name {
            #![allow(unused_imports)]

            use std::io::timer;
            use libc;
            use std::str;
            use std::io::process::Command;
            use native;
            use super::{sleeper, test_destroy_actually_kills};

            fn f() $b

            $($a)* #[test] fn green() { f() }
            $($a)* #[test] fn native() {
                use native;
                let (tx, rx) = channel();
                native::task::spawn(proc() { tx.send(f()) });
                rx.recv();
            }
        }
    )
)

#[cfg(test)] #[start]
fn start(argc: int, argv: *const *const u8) -> int {
    green::start(argc, argv, rustuv::event_loop, test_main)
}

iotest!(fn test_destroy_once() {
    let mut p = sleeper();
    match p.signal_exit() {
        Ok(()) => {}
        Err(e) => fail!("error: {}", e),
    }
})

#[cfg(unix)]
pub fn sleeper() -> Process {
    Command::new("sleep").arg("1000").spawn().unwrap()
}
#[cfg(windows)]
pub fn sleeper() -> Process {
    // There's a `timeout` command on windows, but it doesn't like having
    // its output piped, so instead just ping ourselves a few times with
    // gaps in between so we're sure this process is alive for awhile
    Command::new("ping").arg("127.0.0.1").arg("-n").arg("1000").spawn().unwrap()
}

iotest!(fn test_destroy_twice() {
    let mut p = sleeper();
    succeed!(p.signal_exit()); // this shouldnt crash...
    let _ = p.signal_exit(); // ...and nor should this (and nor should the destructor)
})

pub fn test_destroy_actually_kills(force: bool) {
    use std::io::process::{Command, ProcessOutput, ExitStatus, ExitSignal};
    use std::io::timer;
    use libc;
    use std::str;

    #[cfg(unix,not(target_os="android"))]
    static BLOCK_COMMAND: &'static str = "cat";

    #[cfg(unix,target_os="android")]
    static BLOCK_COMMAND: &'static str = "/system/bin/cat";

    #[cfg(windows)]
    static BLOCK_COMMAND: &'static str = "cmd";

    // this process will stay alive indefinitely trying to read from stdin
    let mut p = Command::new(BLOCK_COMMAND).spawn().unwrap();

    assert!(p.signal(0).is_ok());

    if force {
        p.signal_kill().unwrap();
    } else {
        p.signal_exit().unwrap();
    }

    // Don't let this test time out, this should be quick
    let (tx, rx1) = channel();
    let mut t = timer::Timer::new().unwrap();
    let rx2 = t.oneshot(Duration::milliseconds(1000));
    spawn(proc() {
        select! {
            () = rx2.recv() => unsafe { libc::exit(1) },
            () = rx1.recv() => {}
        }
    });
    match p.wait().unwrap() {
        ExitStatus(..) => fail!("expected a signal"),
        ExitSignal(..) => tx.send(()),
    }
}

iotest!(fn test_unforced_destroy_actually_kills() {
    test_destroy_actually_kills(false);
})

iotest!(fn test_forced_destroy_actually_kills() {
    test_destroy_actually_kills(true);
})
