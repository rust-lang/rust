// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use fmt;
use libc;
use option::{Option, Some, None};
use result::{Ok, Err};
use rt::rtio::{IoFactory, RtioTTY, with_local_io};
use super::{Reader, Writer, io_error};

/// Creates a new non-blocking handle to the stdin of the current process.
///
/// See `stdout()` for notes about this function.
pub fn stdin() -> StdReader {
    do with_local_io |io| {
        match io.tty_open(libc::STDIN_FILENO, true, false) {
            Ok(tty) => Some(StdReader { inner: tty }),
            Err(e) => {
                io_error::cond.raise(e);
                None
            }
        }
    }.unwrap()
}

/// Creates a new non-blocking handle to the stdout of the current process.
///
/// Note that this is a fairly expensive operation in that at least one memory
/// allocation is performed. Additionally, this must be called from a runtime
/// task context because the stream returned will be a non-blocking object using
/// the local scheduler to perform the I/O.
pub fn stdout() -> StdWriter {
    do with_local_io |io| {
        match io.tty_open(libc::STDOUT_FILENO, false, false) {
            Ok(tty) => Some(StdWriter { inner: tty }),
            Err(e) => {
                io_error::cond.raise(e);
                None
            }
        }
    }.unwrap()
}

/// Creates a new non-blocking handle to the stderr of the current process.
///
/// See `stdout()` for notes about this function.
pub fn stderr() -> StdWriter {
    do with_local_io |io| {
        match io.tty_open(libc::STDERR_FILENO, false, false) {
            Ok(tty) => Some(StdWriter { inner: tty }),
            Err(e) => {
                io_error::cond.raise(e);
                None
            }
        }
    }.unwrap()
}

/// Prints a string to the stdout of the current process. No newline is emitted
/// after the string is printed.
pub fn print(s: &str) {
    // XXX: need to see if not caching stdin() is the cause of performance
    //      issues, it should be possible to cache a stdout handle in each Task
    //      and then re-use that across calls to print/println. Note that the
    //      resolution of this comment will affect all of the prints below as
    //      well.
    stdout().write(s.as_bytes());
}

/// Prints a string as a line. to the stdout of the current process. A literal
/// `\n` character is printed to the console after the string.
pub fn println(s: &str) {
    let mut out = stdout();
    out.write(s.as_bytes());
    out.write(['\n' as u8]);
}

/// Similar to `print`, but takes a `fmt::Arguments` structure to be compatible
/// with the `format_args!` macro.
pub fn print_args(fmt: &fmt::Arguments) {
    let mut out = stdout();
    fmt::write(&mut out as &mut Writer, fmt);
}

/// Similar to `println`, but takes a `fmt::Arguments` structure to be
/// compatible with the `format_args!` macro.
pub fn println_args(fmt: &fmt::Arguments) {
    let mut out = stdout();
    fmt::writeln(&mut out as &mut Writer, fmt);
}

/// Representation of a reader of a standard input stream
pub struct StdReader {
    priv inner: ~RtioTTY
}

impl StdReader {
    /// Controls whether this output stream is a "raw stream" or simply a normal
    /// stream.
    ///
    /// # Failure
    ///
    /// This function will raise on the `io_error` condition if an error
    /// happens.
    pub fn set_raw(&mut self, raw: bool) {
        match self.inner.set_raw(raw) {
            Ok(()) => {},
            Err(e) => io_error::cond.raise(e),
        }
    }

    /// Resets the mode of this stream back to its original state.
    ///
    /// # Failure
    ///
    /// This function cannot fail.
    pub fn reset_mode(&mut self) { self.inner.reset_mode(); }
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
    priv inner: ~RtioTTY
}

impl StdWriter {
    /// Gets the size of this output window, if possible. This is typically used
    /// when the writer is attached to something like a terminal, this is used
    /// to fetch the dimensions of the terminal.
    ///
    /// If successful, returns Some((width, height)).
    ///
    /// # Failure
    ///
    /// This function will raise on the `io_error` condition if an error
    /// happens.
    pub fn winsize(&mut self) -> Option<(int, int)> {
        match self.inner.get_winsize() {
            Ok(p) => Some(p),
            Err(e) => {
                io_error::cond.raise(e);
                None
            }
        }
    }

    /// Controls whether this output stream is a "raw stream" or simply a normal
    /// stream.
    ///
    /// # Failure
    ///
    /// This function will raise on the `io_error` condition if an error
    /// happens.
    pub fn set_raw(&mut self, raw: bool) {
        match self.inner.set_raw(raw) {
            Ok(()) => {},
            Err(e) => io_error::cond.raise(e),
        }
    }

    /// Resets the mode of this stream back to its original state.
    ///
    /// # Failure
    ///
    /// This function cannot fail.
    pub fn reset_mode(&mut self) { self.inner.reset_mode(); }
}

impl Writer for StdWriter {
    fn write(&mut self, buf: &[u8]) {
        match self.inner.write(buf) {
            Ok(()) => {}
            Err(e) => io_error::cond.raise(e)
        }
    }

    fn flush(&mut self) { /* nothing to do */ }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn smoke() {
        // Just make sure we can acquire handles
        stdin();
        stdout();
        stderr();
    }
}
