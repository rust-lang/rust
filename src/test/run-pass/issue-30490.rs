// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Previously libstd would set stdio descriptors of a child process
// by `dup`ing the requested descriptors to inherit directly into the
// stdio descriptors. This, however, would incorrectly handle cases
// where the descriptors to inherit were already stdio descriptors.
// This test checks to avoid that regression.

#![cfg_attr(unix, feature(libc))]
#![cfg_attr(windows, allow(unused_imports))]

#[cfg(unix)]
extern crate libc;

use std::fs::File;
use std::io::{Read, Write};
use std::io::{stdout, stderr};
use std::process::{Command, Stdio};

#[cfg(unix)]
use std::os::unix::io::FromRawFd;

#[cfg(not(unix))]
fn main() {
    // Bug not present in Windows
}

#[cfg(unix)]
fn main() {
    let mut args = std::env::args();
    let name = args.next().unwrap();
    let args: Vec<String> = args.collect();
    if let Some("--child") = args.get(0).map(|s| &**s) {
        return child();
    } else if !args.is_empty() {
        panic!("unknown options");
    }

    let stdout_backup = unsafe { libc::dup(libc::STDOUT_FILENO) };
    let stderr_backup = unsafe { libc::dup(libc::STDERR_FILENO) };
    assert!(stdout_backup > -1);
    assert!(stderr_backup > -1);

    let (stdout_reader, stdout_writer) = pipe();
    let (stderr_reader, stderr_writer) = pipe();
    assert!(unsafe { libc::dup2(stdout_writer, libc::STDOUT_FILENO) } > -1);
    assert!(unsafe { libc::dup2(stderr_writer, libc::STDERR_FILENO) } > -1);

    // Make sure we close any duplicates of the writer end of the pipe,
    // otherwise we can get stuck reading from the pipe which has open
    // writers but no one supplying any input
    assert_eq!(unsafe { libc::close(stdout_writer) }, 0);
    assert_eq!(unsafe { libc::close(stderr_writer) }, 0);

    stdout().write_all("parent stdout\n".as_bytes()).expect("failed to write to stdout");
    stderr().write_all("parent stderr\n".as_bytes()).expect("failed to write to stderr");

    let child = {
        Command::new(name)
            .arg("--child")
            .stdin(Stdio::inherit())
            .stdout(unsafe { FromRawFd::from_raw_fd(libc::STDERR_FILENO) })
            .stderr(unsafe { FromRawFd::from_raw_fd(libc::STDOUT_FILENO) })
            .spawn()
    };

    // The Stdio passed into the Command took over (and closed) std{out, err}
    // so we should restore them as they were.
    assert!(unsafe { libc::dup2(stdout_backup, libc::STDOUT_FILENO) } > -1);
    assert!(unsafe { libc::dup2(stderr_backup, libc::STDERR_FILENO) } > -1);

    // Using File as a shim around the descriptor
    let mut read = String::new();
    let mut f: File = unsafe { FromRawFd::from_raw_fd(stdout_reader) };
    f.read_to_string(&mut read).expect("failed to read from stdout file");
    assert_eq!(read, "parent stdout\nchild stderr\n");

    // Using File as a shim around the descriptor
    read.clear();
    let mut f: File = unsafe { FromRawFd::from_raw_fd(stderr_reader) };
    f.read_to_string(&mut read).expect("failed to read from stderr file");
    assert_eq!(read, "parent stderr\nchild stdout\n");

    assert!(child.expect("failed to execute child process").wait().unwrap().success());
}

#[cfg(unix)]
fn child() {
    stdout().write_all("child stdout\n".as_bytes()).expect("child failed to write to stdout");
    stderr().write_all("child stderr\n".as_bytes()).expect("child failed to write to stderr");
}

#[cfg(unix)]
/// Returns a pipe (reader, writer combo)
fn pipe() -> (i32, i32) {
     let mut fds = [0; 2];
     assert_eq!(unsafe { libc::pipe(fds.as_mut_ptr()) }, 0);
     (fds[0], fds[1])
}
