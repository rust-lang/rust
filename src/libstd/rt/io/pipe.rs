// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Synchronous, in-memory pipes.
//!
//! Currently these aren't particularly useful, there only exists bindings
//! enough so that pipes can be created to child processes.

use prelude::*;
use super::{Reader, Writer};
use rt::io::{io_error, EndOfFile};
use rt::io::native::file;
use rt::rtio::{RtioPipe, with_local_io};

pub struct PipeStream {
    priv obj: ~RtioPipe,
}

impl PipeStream {
    /// Consumes a file descriptor to return a pipe stream that will have
    /// synchronous, but non-blocking reads/writes. This is useful if the file
    /// descriptor is acquired via means other than the standard methods.
    ///
    /// This operation consumes ownership of the file descriptor and it will be
    /// closed once the object is deallocated.
    ///
    /// # Example
    ///
    ///     use std::libc;
    ///     use std::rt::io::pipe;
    ///
    ///     let mut pipe = PipeStream::open(libc::STDERR_FILENO);
    ///     pipe.write(bytes!("Hello, stderr!"));
    ///
    /// # Failure
    ///
    /// If the pipe cannot be created, an error will be raised on the
    /// `io_error` condition.
    pub fn open(fd: file::fd_t) -> Option<PipeStream> {
        do with_local_io |io| {
            match io.pipe_open(fd) {
                Ok(obj) => Some(PipeStream { obj: obj }),
                Err(e) => {
                    io_error::cond.raise(e);
                    None
                }
            }
        }
    }

    pub fn new(inner: ~RtioPipe) -> PipeStream {
        PipeStream { obj: inner }
    }
}

impl Reader for PipeStream {
    fn read(&mut self, buf: &mut [u8]) -> Option<uint> {
        match self.obj.read(buf) {
            Ok(read) => Some(read),
            Err(ioerr) => {
                // EOF is indicated by returning None
                if ioerr.kind != EndOfFile {
                    io_error::cond.raise(ioerr);
                }
                return None;
            }
        }
    }

    fn eof(&mut self) -> bool { false }
}

impl Writer for PipeStream {
    fn write(&mut self, buf: &[u8]) {
        match self.obj.write(buf) {
            Ok(_) => (),
            Err(ioerr) => {
                io_error::cond.raise(ioerr);
            }
        }
    }

    fn flush(&mut self) {}
}
