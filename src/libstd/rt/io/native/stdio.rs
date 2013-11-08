// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use libc;
use option::Option;
use rt::io::{Reader, Writer};
use super::file;

/// Creates a new handle to the stdin of this process
pub fn stdin() -> StdIn { StdIn::new() }
/// Creates a new handle to the stdout of this process
pub fn stdout() -> StdOut { StdOut::new(libc::STDOUT_FILENO) }
/// Creates a new handle to the stderr of this process
pub fn stderr() -> StdOut { StdOut::new(libc::STDERR_FILENO) }

pub fn print(s: &str) {
    stdout().write(s.as_bytes())
}

pub fn println(s: &str) {
    let mut out = stdout();
    out.write(s.as_bytes());
    out.write(['\n' as u8]);
}

pub struct StdIn {
    priv fd: file::FileDesc
}

impl StdIn {
    /// Duplicates the stdin file descriptor, returning an io::Reader
    pub fn new() -> StdIn {
        StdIn { fd: file::FileDesc::new(libc::STDIN_FILENO, false) }
    }
}

impl Reader for StdIn {
    fn read(&mut self, buf: &mut [u8]) -> Option<uint> { self.fd.read(buf) }
    fn eof(&mut self) -> bool { self.fd.eof() }
}

pub struct StdOut {
    priv fd: file::FileDesc
}

impl StdOut {
    /// Duplicates the specified file descriptor, returning an io::Writer
    pub fn new(fd: file::fd_t) -> StdOut {
        StdOut { fd: file::FileDesc::new(fd, false) }
    }
}

impl Writer for StdOut {
    fn write(&mut self, buf: &[u8]) { self.fd.write(buf) }
    fn flush(&mut self) { self.fd.flush() }
}
