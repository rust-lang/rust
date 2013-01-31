// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use core::io::{Reader, BytesReader};
use core::io;
use core::prelude::*;

pub struct BufReader {
    buf: ~[u8],
    mut pos: uint
}

pub impl BufReader {
    static pub fn new(v: ~[u8]) -> BufReader {
        BufReader {
            buf: move v,
            pos: 0
        }
    }

    priv fn as_bytes_reader<A>(f: &fn(&BytesReader) -> A) -> A {
        // Recreating the BytesReader state every call since
        // I can't get the borrowing to work correctly
        let bytes_reader = BytesReader {
            bytes: ::core::util::id::<&[u8]>(self.buf),
            pos: self.pos
        };

        let res = f(&bytes_reader);

        // FIXME #4429: This isn't correct if f fails
        self.pos = bytes_reader.pos;

        return move res;
    }
}

impl BufReader: Reader {
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
