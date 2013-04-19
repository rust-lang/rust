// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Blocking posix-based file I/O

use prelude::*;
use super::super::*;
use libc::{c_int, FILE};

#[allow(non_camel_case_types)]
pub type fd_t = c_int;

// Make this a newtype so we can't do I/O on arbitrary integers
pub struct FileDesc(fd_t);

impl FileDesc {
    /// Create a `FileDesc` from an open C file descriptor.
    ///
    /// The `FileDesc` takes ownership of the file descriptor
    /// and will close it upon destruction.
    pub fn new(_fd: fd_t) -> FileDesc { fail!() }
}

impl Reader for FileDesc {
    fn read(&mut self, _buf: &mut [u8]) -> Option<uint> { fail!() }

    fn eof(&mut self) -> bool { fail!() }
}

impl Writer for FileDesc {
    fn write(&mut self, _buf: &[u8]) { fail!() }

    fn flush(&mut self) { fail!() }
}

impl Close for FileDesc {
    fn close(&mut self) { fail!() }
}

impl Seek for FileDesc {
    fn tell(&self) -> u64 { fail!() }

    fn seek(&mut self, _pos: i64, _style: SeekStyle) { fail!() }
}

pub struct CFile(*FILE);

impl CFile {
    /// Create a `CFile` from an open `FILE` pointer.
    ///
    /// The `CFile` takes ownership of the file descriptor
    /// and will close it upon destruction.
    pub fn new(_file: *FILE) -> CFile { fail!() }
}

impl Reader for CFile {
    fn read(&mut self, _buf: &mut [u8]) -> Option<uint> { fail!() }

    fn eof(&mut self) -> bool { fail!() }
}

impl Writer for CFile {
    fn write(&mut self, _buf: &[u8]) { fail!() }

    fn flush(&mut self) { fail!() }
}

impl Close for CFile {
    fn close(&mut self) { fail!() }
}

impl Seek for CFile {
    fn tell(&self) -> u64 { fail!() }
    fn seek(&mut self, _pos: i64, _style: SeekStyle) { fail!() }
}
