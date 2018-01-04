// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// ignore-emscripten spawning processes is not supported

#![feature(close_std_streams)]

use std::{env, io, process};
use std::io::{Read, Write};

fn expect_read(stream: &mut Read, msg: &[u8]) {
    assert!(msg.len() < 128);

    let mut buf = [0u8; 128];
    match stream.read(&mut buf) {
        Ok(n) => {
            assert_eq!(n, msg.len());
            if n > 0 {
                assert_eq!(&buf[..n], msg);
            }
        },
        Err(e) => panic!("read error: {}", e)
    }
}

// Note: order of operations is critical in this program.
// Parent and child are synchronized by which streams are closed when.

fn child_locked() {
    let     stdin_u  = io::stdin();
    let     stdout_u = io::stdout();
    let     stderr_u = io::stderr();
    let mut stdin    = stdin_u.lock();
    let mut stdout   = stdout_u.lock();
    let mut stderr   = stderr_u.lock();

    expect_read(&mut stdin, b"in1\n");
    stdin.close().unwrap();

    stdout.write(b"ou1\n").unwrap();
    stdout.close().unwrap();
    stdout.write(b"ou2\n").unwrap();

    expect_read(&mut stdin, b"");

    // stderr is tested last, because we will lose the ability to emit
    // panic messages when we close stderr.
    stderr.write(b"er1\n").unwrap();
    stderr.close().unwrap();
    stderr.write(b"er2\n").unwrap();
}

fn child_unlocked() {
    expect_read(&mut io::stdin(), b"in1\n");
    io::stdin().close().unwrap();

    io::stdout().write(b"ou1\n").unwrap();
    io::stdout().close().unwrap();
    io::stdout().write(b"ou2\n").unwrap();

    expect_read(&mut io::stdin(), b"");

    // stderr is tested last, because we will lose the ability to emit
    // panic messages when we close stderr.
    io::stderr().write(b"er1\n").unwrap();
    io::stderr().close().unwrap();
    io::stderr().write(b"er2\n").unwrap();
}

fn parent(arg: &'static str) {
    let this = env::args().next().unwrap();
    let mut child = process::Command::new(this)
        .arg(arg)
        .stdin(process::Stdio::piped())
        .stdout(process::Stdio::piped())
        .stderr(process::Stdio::piped())
        .spawn()
        .unwrap();

    let mut c_stdin = child.stdin.take().unwrap();
    let mut c_stdout = child.stdout.take().unwrap();
    let mut c_stderr = child.stderr.take().unwrap();

    // this will be received by the child
    c_stdin.write(b"in1\n").unwrap();

    // reading this also synchronizes with the child closing its stdin
    expect_read(&mut c_stdout, b"ou1\n");

    // this should signal a broken pipe
    match c_stdin.write(b"in2\n") {
        Ok(_) => panic!("second write to child should not have succeeded"),
        Err(e) => {
            if e.kind() != io::ErrorKind::BrokenPipe {
                panic!("second write to child failed the wrong way: {}", e)
            }
        }
    }

    expect_read(&mut c_stdout, b"");
    expect_read(&mut c_stderr, b"er1\n");
    expect_read(&mut c_stderr, b"");

    let status = child.wait().unwrap();
    assert!(status.success());
}

fn main() {
    let n = env::args().count();
    if n == 1 {
        parent("L");
        parent("U");
    } else if n == 2 {
        match env::args().nth(1).unwrap().as_ref() {
            "L" => child_locked(),
            "U" => child_unlocked(),
            s => panic!("child selector {} not recognized", s)
        }
    } else {
        panic!("wrong number of arguments - {}", n)
    }
}
