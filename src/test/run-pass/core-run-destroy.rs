// Copyright 2012-2013-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// compile-flags:--test
// ignore-emscripten

// NB: These tests kill child processes. Valgrind sees these children as leaking
// memory, which makes for some *confusing* logs. That's why these are here
// instead of in std.

#![reexport_test_harness_main = "test_main"]
#![feature(libc, std_misc, duration)]

extern crate libc;

use std::process::{self, Command, Child, Output, Stdio};
use std::str;
use std::sync::mpsc::channel;
use std::thread;
use std::time::Duration;

macro_rules! t {
    ($e:expr) => (match $e { Ok(e) => e, Err(e) => panic!("error: {}", e) })
}

#[test]
fn test_destroy_once() {
    let mut p = sleeper();
    t!(p.kill());
}

#[cfg(unix)]
pub fn sleeper() -> Child {
    t!(Command::new("sleep").arg("1000").spawn())
}
#[cfg(windows)]
pub fn sleeper() -> Child {
    // There's a `timeout` command on windows, but it doesn't like having
    // its output piped, so instead just ping ourselves a few times with
    // gaps in between so we're sure this process is alive for awhile
    t!(Command::new("ping").arg("127.0.0.1").arg("-n").arg("1000").spawn())
}

#[test]
fn test_destroy_twice() {
    let mut p = sleeper();
    t!(p.kill()); // this shouldn't crash...
    let _ = p.kill(); // ...and nor should this (and nor should the destructor)
}

#[test]
fn test_destroy_actually_kills() {
    let cmd = if cfg!(windows) {
        "cmd"
    } else if cfg!(target_os = "android") {
        "/system/bin/cat"
    } else {
        "cat"
    };

    // this process will stay alive indefinitely trying to read from stdin
    let mut p = t!(Command::new(cmd)
                           .stdin(Stdio::piped())
                           .spawn());

    t!(p.kill());

    // Don't let this test time out, this should be quick
    let (tx, rx) = channel();
    thread::spawn(move|| {
        thread::sleep_ms(1000);
        if rx.try_recv().is_err() {
            process::exit(1);
        }
    });
    let code = t!(p.wait()).code();
    if cfg!(windows) {
        assert!(code.is_some());
    } else {
        assert!(code.is_none());
    }
    tx.send(());
}
