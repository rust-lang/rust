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
use option::{Option, Some, None};
use result::{Ok, Err};
use rt::local::Local;
use rt::rtio::{RtioFileStream, IoFactoryObject, IoFactory};
use super::{Reader, Writer, io_error};

/// Creates a new non-blocking handle to the stdin of the current process.
///
/// See `stdout()` for notes about this function.
pub fn stdin() -> StdReader {
    let stream = unsafe {
        let io: *mut IoFactoryObject = Local::unsafe_borrow();
        (*io).fs_from_raw_fd(libc::STDIN_FILENO, false)
    };
    StdReader { inner: stream }
}

/// Creates a new non-blocking handle to the stdout of the current process.
///
/// Note that this is a fairly expensive operation in that at least one memory
/// allocation is performed. Additionally, this must be called from a runtime
/// task context because the stream returned will be a non-blocking object using
/// the local scheduler to perform the I/O.
pub fn stdout() -> StdWriter {
    let stream = unsafe {
        let io: *mut IoFactoryObject = Local::unsafe_borrow();
        (*io).fs_from_raw_fd(libc::STDOUT_FILENO, false)
    };
    StdWriter { inner: stream }
}

/// Creates a new non-blocking handle to the stderr of the current process.
///
/// See `stdout()` for notes about this function.
pub fn stderr() -> StdWriter {
    let stream = unsafe {
        let io: *mut IoFactoryObject = Local::unsafe_borrow();
        (*io).fs_from_raw_fd(libc::STDERR_FILENO, false)
    };
    StdWriter { inner: stream }
}

/// Prints a string to the stdout of the current process. No newline is emitted
/// after the string is printed.
pub fn print(s: &str) {
    // XXX: need to see if not caching stdin() is the cause of performance
    //      issues, it should be possible to cache a stdout handle in each Task
    //      and then re-use that across calls to print/println
    stdout().write(s.as_bytes());
}

/// Prints a string as a line. to the stdout of the current process. A literal
/// `\n` character is printed to the console after the string.
pub fn println(s: &str) {
    let mut out = stdout();
    out.write(s.as_bytes());
    out.write(['\n' as u8]);
}

/// Representation of a reader of a standard input stream
pub struct StdReader {
    priv inner: ~RtioFileStream
}

impl Reader for StdReader {
    fn read(&mut self, buf: &mut [u8]) -> Option<uint> {
        match self.inner.read(buf) {
            Ok(amt) => Some(amt as uint),
            Err(e) => {
                io_error::cond.raise(e);
                None
            }
        }
    }

    fn eof(&mut self) -> bool { false }
}

/// Representation of a writer to a standard output stream
pub struct StdWriter {
    priv inner: ~RtioFileStream
}

impl Writer for StdWriter {
    fn write(&mut self, buf: &[u8]) {
        match self.inner.write(buf) {
            Ok(()) => {}
            Err(e) => io_error::cond.raise(e)
        }
    }

    fn flush(&mut self) {
        match self.inner.flush() {
            Ok(()) => {}
            Err(e) => io_error::cond.raise(e)
        }
    }
}
