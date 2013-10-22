// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#[allow(missing_doc)];

use std::io::{Reader, BytesReader};
use std::io;
use std::cast;

/// An implementation of the io::Reader interface which reads a buffer of bytes
pub struct BufReader {
    /// The buffer of bytes to read
    priv buf: ~[u8],
    /// The current position in the buffer of bytes
    priv pos: @mut uint
}

impl BufReader {
    /// Creates a new buffer reader for the specified buffer
    pub fn new(v: ~[u8]) -> BufReader {
        BufReader {
            buf: v,
            pos: @mut 0
        }
    }

    fn as_bytes_reader<A>(&self, f: &fn(&BytesReader) -> A) -> A {
        // FIXME(#5723)
        let bytes = ::std::util::id::<&[u8]>(self.buf);
        let bytes: &'static [u8] = unsafe { cast::transmute(bytes) };
        // Recreating the BytesReader state every call since
        // I can't get the borrowing to work correctly
        let bytes_reader = BytesReader {
            bytes: bytes,
            pos: @mut *self.pos
        };

        let res = f(&bytes_reader);

        // FIXME #4429: This isn't correct if f fails
        *self.pos = *bytes_reader.pos;

        return res;
    }
}

impl Reader for BufReader {
    fn read(&self, bytes: &mut [u8], len: uint) -> uint {
        self.as_bytes_reader(|r| r.read(bytes, len) )
    }
    fn read_byte(&self) -> int {
        self.as_bytes_reader(|r| r.read_byte() )
    }
    fn eof(&self) -> bool {
        self.as_bytes_reader(|r| r.eof() )
    }
    fn seek(&self, offset: int, whence: io::SeekStyle) {
        self.as_bytes_reader(|r| r.seek(offset, whence) )
    }
    fn tell(&self) -> uint {
        self.as_bytes_reader(|r| r.tell() )
    }
}
