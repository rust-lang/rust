// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Readers and Writers using channels as the underlying transport
//!
//! This module is useful for communication between tasks through Readers and
//! Writers. This implements the Reader/Writer interface on top of the Port/Chan
//! types to provide a blocking `read` method and an always-asynchronous `write`
//! method.
//!
//! # Example
//!
//!     use std::io::comm;
//!
//!     let (mut reader, writer) = comm::stream();
//!
//!     do spawn {
//!         let mut writer = writer;
//!         writer.write([1, 2, 3]);
//!     }
//!
//!     assert_eq!(reader.read_to_end(), ~[1, 2, 3]);

use cmp;
use comm;
use comm::{GenericChan, GenericPort, Port, Chan};
use io::{Writer, Reader};
use option::{Some, None, Option};
use vec;
use vec::{ImmutableVector, CopyableVector};

/// Creates a new connected (reader, writer) pair. All data written on the
/// writer will show up on the reader.
pub fn stream() -> (PortReader, ChanWriter) {
    let (port, chan) = comm::stream();
    (PortReader { prev: None, port: port, pos: 0 }, ChanWriter { chan: chan })
}

/// Wrapper struct around an internal communication port to read data from
pub struct PortReader {
    priv prev: Option<~[u8]>,
    priv pos: uint,
    priv port: Port<~[u8]>,
}

/// Wrapper struct around a channel to implement a `Writer` interface.
pub struct ChanWriter {
    priv chan: Chan<~[u8]>,
}

impl Writer for ChanWriter {
    fn write(&mut self, data: &[u8]) {
        self.chan.send(data.to_owned());
    }
}

impl Reader for PortReader {
    fn read(&mut self, data: &mut [u8]) -> Option<uint> {
        let mut offset = 0;
        while offset < data.len() {
            match self.prev {
                Some(ref b) => {
                    let dst = data.mut_slice_from(offset);
                    let src = b.slice_from(self.pos);
                    let amt = cmp::min(dst.len(), src.len());
                    vec::bytes::copy_memory(dst, src, amt);
                    self.pos += amt;
                    offset += amt;
                    if self.pos < b.len() {
                        break
                    }
                }
                None => {}
            }
            self.prev = self.port.try_recv();
            self.pos = 0;
            if self.prev.is_none() { break }
        }
        if offset == 0 {
            None
        } else {
            Some(offset)
        }
    }

    fn eof(&mut self) -> bool { false }
}

#[cfg(test)]
mod tests {
    use prelude::*;

    use io::comm::stream;
    use util;

    #[test]
    fn smoke() {
        let (mut reader, mut writer) = stream();
        writer.write([1, 2, 3, 4]);
        util::ignore(writer);

        assert_eq!(reader.read_byte(), Some(1));
        let mut buf = [0, 0];
        assert_eq!(reader.read(buf), Some(2));
        assert_eq!(buf[0], 2);
        assert_eq!(buf[1], 3);

        assert_eq!(reader.read(buf), Some(1));
        assert_eq!(buf[0], 4);

        assert_eq!(reader.read(buf), None);
    }

    #[test]
    fn parallel() {
        let (mut reader, writer) = stream();
        do spawn {
            let mut writer = writer;
            writer.write([1, 2]);
            writer.write([3]);
            writer.write([4]);
        }

        assert_eq!(reader.read_byte(), Some(1));
        let mut buf = [0, 0];
        assert_eq!(reader.read(buf), Some(2));
        assert_eq!(buf[0], 2);
        assert_eq!(buf[1], 3);

        assert_eq!(reader.read(buf), Some(1));
        assert_eq!(buf[0], 4);

        assert_eq!(reader.read(buf), None);
    }
}
