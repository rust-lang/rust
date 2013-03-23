// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

pub mod file;

// FIXME #5370 Strongly want this to be StreamError(&mut Stream)
pub struct StreamError;

// XXX: Can't put doc comments on macros
// Raised by `Stream` instances on error. Returning `true` from the handler
// indicates that the `Stream` should continue, `false` that it should fail.
condition! {
    stream_error: super::StreamError -> bool;
}

pub trait Stream {
    /// Read bytes, up to the length of `buf` and place them in `buf`,
    /// returning the number of bytes read or an `IoError`. Reads
    /// 0 bytes on EOF.
    ///
    /// # Failure
    ///
    /// Raises the `reader_error` condition on error
    fn read(&mut self, buf: &mut [u8]) -> uint;

    /// Return whether the Reader has reached the end of the stream
    fn eof(&mut self) -> bool;

    /// Write the given buffer
    ///
    /// # Failure
    ///
    /// Raises the `writer_error` condition on error
    fn write(&mut self, v: &const [u8]);
}
