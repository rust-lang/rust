// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// ignore-win32
// ignore-android

// This test ensures that the 'detach' field on processes does the right thing.
// By detaching the child process, they should be put into a separate process
// group. We test this by spawning a detached process, then killing our own
// group with a signal.
//
// Note that the first thing we do is put ourselves in our own process group so
// we don't interfere with other running tests.

extern crate green;
extern crate rustuv;
extern crate libc;

use std::io::process;
use std::io::signal::{Listener, Interrupt};

#[start]
fn start(argc: int, argv: **u8) -> int {
    green::start(argc, argv, rustuv::event_loop, main)
}

fn main() {
    unsafe { libc::setsid(); }

    let config = process::ProcessConfig {
        program : "/bin/sh",
        args: &["-c".to_owned(), "read a".to_owned()],
        detach: true,
        .. process::ProcessConfig::new()
    };

    // we shouldn't die because of an interrupt
    let mut l = Listener::new();
    l.register(Interrupt).unwrap();

    // spawn the child
    let mut p = process::Process::configure(config).unwrap();

    // send an interrupt to everyone in our process group
    unsafe { libc::funcs::posix88::signal::kill(0, libc::SIGINT); }

    // Wait for the child process to die (terminate it's stdin and the read
    // should fail).
    drop(p.stdin.take());
    match p.wait() {
        process::ExitStatus(..) => {}
        process::ExitSignal(..) => fail!()
    }
}

