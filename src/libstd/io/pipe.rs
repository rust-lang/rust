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

#![allow(missing_doc)]

use prelude::*;
use io::{IoResult, IoError};
use libc;
use owned::Box;
use rt::rtio::{RtioPipe, LocalIo};

/// A synchronous, in-memory pipe.
pub struct PipeStream {
    /// The internal, opaque runtime pipe object.
    obj: Box<RtioPipe:Send>,
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
    /// ```rust
    /// # #![allow(unused_must_use)]
    /// extern crate libc;
    ///
    /// use std::io::pipe::PipeStream;
    ///
    /// fn main() {
    ///     let mut pipe = PipeStream::open(libc::STDERR_FILENO);
    ///     pipe.write(bytes!("Hello, stderr!"));
    /// }
    /// ```
    pub fn open(fd: libc::c_int) -> IoResult<PipeStream> {
        LocalIo::maybe_raise(|io| {
            io.pipe_open(fd).map(|obj| PipeStream { obj: obj })
        }).map_err(IoError::from_rtio_error)
    }

    #[doc(hidden)]
    pub fn new(inner: Box<RtioPipe:Send>) -> PipeStream {
        PipeStream { obj: inner }
    }
}

impl Clone for PipeStream {
    fn clone(&self) -> PipeStream {
        PipeStream { obj: self.obj.clone() }
    }
}

impl Reader for PipeStream {
    fn read(&mut self, buf: &mut [u8]) -> IoResult<uint> {
        self.obj.read(buf).map_err(IoError::from_rtio_error)
    }
}

impl Writer for PipeStream {
    fn write(&mut self, buf: &[u8]) -> IoResult<()> {
        self.obj.write(buf).map_err(IoError::from_rtio_error)
    }
}

#[cfg(test)]
mod test {
    iotest!(fn partial_read() {
        use os;
        use io::pipe::PipeStream;

        let os::Pipe { input, out } = os::pipe();
        let out = PipeStream::open(out);
        let mut input = PipeStream::open(input);
        let (tx, rx) = channel();
        spawn(proc() {
            let mut out = out;
            out.write([10]).unwrap();
            rx.recv(); // don't close the pipe until the other read has finished
        });

        let mut buf = [0, ..10];
        input.read(buf).unwrap();
        tx.send(());
    })
}
