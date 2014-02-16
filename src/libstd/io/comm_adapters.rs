// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use prelude::*;

use comm::{Port, Chan};
use cmp;
use io;
use option::{None, Option, Some};
use super::{Reader, Writer, IoResult};
use vec::{bytes, CloneableVector, MutableVector, ImmutableVector};

/// Allows reading from a port.
///
/// # Example
///
/// ```
/// use std::io::PortReader;
///
/// let (port, chan) = Chan::new();
/// # drop(chan);
/// let mut reader = PortReader::new(port);
///
/// let mut buf = ~[0u8, ..100];
/// match reader.read(buf) {
///     Ok(nread) => println!("Read {} bytes", nread),
///     Err(e) => println!("read error: {}", e),
/// }
/// ```
pub struct PortReader {
    priv buf: Option<~[u8]>,  // A buffer of bytes received but not consumed.
    priv pos: uint,           // How many of the buffered bytes have already be consumed.
    priv port: Port<~[u8]>,   // The port to pull data from.
    priv closed: bool,        // Whether the pipe this port connects to has been closed.
}

impl PortReader {
    pub fn new(port: Port<~[u8]>) -> PortReader {
        PortReader {
            buf: None,
            pos: 0,
            port: port,
            closed: false,
        }
    }
}

impl Reader for PortReader {
    fn read(&mut self, buf: &mut [u8]) -> IoResult<uint> {
        let mut num_read = 0;
        loop {
            match self.buf {
                Some(ref prev) => {
                    let dst = buf.mut_slice_from(num_read);
                    let src = prev.slice_from(self.pos);
                    let count = cmp::min(dst.len(), src.len());
                    bytes::copy_memory(dst, src.slice_to(count));
                    num_read += count;
                    self.pos += count;
                },
                None => (),
            };
            if num_read == buf.len() || self.closed {
                break;
            }
            self.pos = 0;
            self.buf = self.port.recv_opt();
            self.closed = self.buf.is_none();
        }
        if self.closed && num_read == 0 {
            Err(io::standard_error(io::EndOfFile))
        } else {
            Ok(num_read)
        }
    }
}

/// Allows writing to a chan.
///
/// # Example
///
/// ```
/// # #[allow(unused_must_use)];
/// use std::io::ChanWriter;
///
/// let (port, chan) = Chan::new();
/// # drop(port);
/// let mut writer = ChanWriter::new(chan);
/// writer.write("hello, world".as_bytes());
/// ```
pub struct ChanWriter {
    chan: Chan<~[u8]>,
}

impl ChanWriter {
    pub fn new(chan: Chan<~[u8]>) -> ChanWriter {
        ChanWriter { chan: chan }
    }
}

impl Clone for ChanWriter {
    fn clone(&self) -> ChanWriter {
        ChanWriter { chan: self.chan.clone() }
    }
}

impl Writer for ChanWriter {
    fn write(&mut self, buf: &[u8]) -> IoResult<()> {
        if !self.chan.try_send(buf.to_owned()) {
            Err(io::IoError {
                kind: io::BrokenPipe,
                desc: "Pipe closed",
                detail: None
            })
        } else {
            Ok(())
        }
    }
}


#[cfg(test)]
mod test {
    use prelude::*;
    use super::*;
    use io;
    use task;

    #[test]
    fn test_port_reader() {
        let (port, chan) = Chan::new();
        task::spawn(proc() {
          chan.send(~[1u8, 2u8]);
          chan.send(~[]);
          chan.send(~[3u8, 4u8]);
          chan.send(~[5u8, 6u8]);
          chan.send(~[7u8, 8u8]);
        });

        let mut reader = PortReader::new(port);
        let mut buf = ~[0u8, ..3];


        assert_eq!(Ok(0), reader.read([]));

        assert_eq!(Ok(3), reader.read(buf));
        assert_eq!(~[1,2,3], buf);

        assert_eq!(Ok(3), reader.read(buf));
        assert_eq!(~[4,5,6], buf);

        assert_eq!(Ok(2), reader.read(buf));
        assert_eq!(~[7,8,6], buf);

        match reader.read(buf) {
            Ok(..) => fail!(),
            Err(e) => assert_eq!(e.kind, io::EndOfFile),
        }
        assert_eq!(~[7,8,6], buf);

        // Ensure it continues to fail in the same way.
        match reader.read(buf) {
            Ok(..) => fail!(),
            Err(e) => assert_eq!(e.kind, io::EndOfFile),
        }
        assert_eq!(~[7,8,6], buf);
    }

    #[test]
    fn test_chan_writer() {
        let (port, chan) = Chan::new();
        let mut writer = ChanWriter::new(chan);
        writer.write_be_u32(42).unwrap();

        let wanted = ~[0u8, 0u8, 0u8, 42u8];
        let got = task::try(proc() { port.recv() }).unwrap();
        assert_eq!(wanted, got);

        match writer.write_u8(1) {
            Ok(..) => fail!(),
            Err(e) => assert_eq!(e.kind, io::BrokenPipe),
        }
    }
}
