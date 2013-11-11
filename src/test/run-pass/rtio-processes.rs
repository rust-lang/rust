// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// compile-flags: --test
// xfail-fast

// In the current state of affairs, libuv registers a SIGCHLD handler when a
// process is spawned through it. This is not done with a SA_RESTART flag,
// meaning that all of our syscalls run the risk of returning EINTR. This error
// is not correctly handled in the majority of std::io, so these can't run with
// the main body of tests there.
//
// That being said, libuv correctly handles EINTR completely, so these tests
// themselves are safe against that. Currently the test runner may run into this
// problem, but it's less likely than a whole suite of tests...
//
// See #9341

use std::rt::io;
use std::rt::io::process::{Process, ProcessConfig, CreatePipe, Ignored};
use std::str;

#[test]
// FIXME(#10380)
#[cfg(unix, not(target_os="android"))]
fn smoke() {
    let io = ~[];
    let args = ProcessConfig {
        program: "/bin/sh",
        args: [~"-c", ~"true"],
        env: None,
        cwd: None,
        io: io,
    };
    let p = Process::new(args);
    assert!(p.is_some());
    let mut p = p.unwrap();
    assert_eq!(p.wait(), 0);
}

#[test]
// FIXME(#10380)
#[cfg(unix, not(target_os="android"))]
fn smoke_failure() {
    let io = ~[];
    let args = ProcessConfig {
        program: "if-this-is-a-binary-then-the-world-has-ended",
        args: [],
        env: None,
        cwd: None,
        io: io,
    };
    match io::result(|| Process::new(args)) {
        Ok(*) => fail!(),
        Err(*) => {}
    }
}

#[test]
// FIXME(#10380)
#[cfg(unix, not(target_os="android"))]
fn exit_reported_right() {
    let io = ~[];
    let args = ProcessConfig {
        program: "/bin/sh",
        args: [~"-c", ~"exit 1"],
        env: None,
        cwd: None,
        io: io,
    };
    let p = Process::new(args);
    assert!(p.is_some());
    let mut p = p.unwrap();
    assert_eq!(p.wait(), 1);
}

fn read_all(input: &mut Reader) -> ~str {
    let mut ret = ~"";
    let mut buf = [0, ..1024];
    loop {
        match input.read(buf) {
            None => { break }
            Some(n) => { ret = ret + str::from_utf8(buf.slice_to(n)); }
        }
    }
    return ret;
}

fn run_output(args: ProcessConfig) -> ~str {
    let p = Process::new(args);
    assert!(p.is_some());
    let mut p = p.unwrap();
    assert!(p.io[0].is_none());
    assert!(p.io[1].is_some());
    let ret = read_all(p.io[1].get_mut_ref() as &mut Reader);
    assert_eq!(p.wait(), 0);
    return ret;
}

#[test]
// FIXME(#10380)
#[cfg(unix, not(target_os="android"))]
fn stdout_works() {
    let io = ~[Ignored, CreatePipe(false, true)];
    let args = ProcessConfig {
        program: "/bin/sh",
        args: [~"-c", ~"echo foobar"],
        env: None,
        cwd: None,
        io: io,
    };
    assert_eq!(run_output(args), ~"foobar\n");
}

#[test]
// FIXME(#10380)
#[cfg(unix, not(target_os="android"))]
fn set_cwd_works() {
    let io = ~[Ignored, CreatePipe(false, true)];
    let cwd = Some("/");
    let args = ProcessConfig {
        program: "/bin/sh",
        args: [~"-c", ~"pwd"],
        env: None,
        cwd: cwd,
        io: io,
    };
    assert_eq!(run_output(args), ~"/\n");
}

#[test]
// FIXME(#10380)
#[cfg(unix, not(target_os="android"))]
fn stdin_works() {
    let io = ~[CreatePipe(true, false),
               CreatePipe(false, true)];
    let args = ProcessConfig {
        program: "/bin/sh",
        args: [~"-c", ~"read line; echo $line"],
        env: None,
        cwd: None,
        io: io,
    };
    let mut p = Process::new(args).expect("didn't create a proces?!");
    p.io[0].get_mut_ref().write("foobar".as_bytes());
    p.io[0] = None; // close stdin;
    let out = read_all(p.io[1].get_mut_ref() as &mut Reader);
    assert_eq!(p.wait(), 0);
    assert_eq!(out, ~"foobar\n");
}
