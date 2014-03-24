// Copyright 2012-2013-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// ignore-fast
// ignore-pretty
// compile-flags:--test

// NB: These tests kill child processes. Valgrind sees these children as leaking
// memory, which makes for some *confusing* logs. That's why these are here
// instead of in std.

#[feature(macro_rules)];

extern crate native;
extern crate green;
extern crate rustuv;

macro_rules! iotest (
    { fn $name:ident() $b:block $($a:attr)* } => (
        mod $name {
            #[allow(unused_imports)];

            use std::io::timer;
            use std::libc;
            use std::str;
            use std::io::process::{Process, ProcessOutput};
            use native;
            use super::*;

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
fn start(argc: int, argv: **u8) -> int {
    green::start(argc, argv, rustuv::event_loop, __test::main)
}

iotest!(fn test_destroy_once() {
    #[cfg(not(target_os="android"))]
    static mut PROG: &'static str = "echo";

    #[cfg(target_os="android")]
    static mut PROG: &'static str = "ls"; // android don't have echo binary

    let mut p = unsafe {Process::new(PROG, []).unwrap()};
    p.signal_exit().unwrap(); // this shouldn't crash (and nor should the destructor)
})

iotest!(fn test_destroy_twice() {
    #[cfg(not(target_os="android"))]
    static mut PROG: &'static str = "echo";
    #[cfg(target_os="android")]
    static mut PROG: &'static str = "ls"; // android don't have echo binary

    let mut p = match unsafe{Process::new(PROG, [])} {
        Ok(p) => p,
        Err(e) => fail!("wut: {}", e),
    };
    p.signal_exit().unwrap(); // this shouldnt crash...
    p.signal_exit().unwrap(); // ...and nor should this (and nor should the destructor)
})

pub fn test_destroy_actually_kills(force: bool) {
    use std::io::process::{Process, ProcessOutput, ExitStatus, ExitSignal};
    use std::io::timer;
    use std::libc;
    use std::str;

    #[cfg(unix,not(target_os="android"))]
    static BLOCK_COMMAND: &'static str = "cat";

    #[cfg(unix,target_os="android")]
    static BLOCK_COMMAND: &'static str = "/system/bin/cat";

    #[cfg(windows)]
    static BLOCK_COMMAND: &'static str = "cmd";

    // this process will stay alive indefinitely trying to read from stdin
    let mut p = Process::new(BLOCK_COMMAND, []).unwrap();

    assert!(p.signal(0).is_ok());

    if force {
        p.signal_kill().unwrap();
    } else {
        p.signal_exit().unwrap();
    }

    // Don't let this test time out, this should be quick
    let (tx, rx1) = channel();
    let mut t = timer::Timer::new().unwrap();
    let rx2 = t.oneshot(1000);
    spawn(proc() {
        select! {
            () = rx2.recv() => unsafe { libc::exit(1) },
            () = rx1.recv() => {}
        }
    });
    match p.wait() {
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
