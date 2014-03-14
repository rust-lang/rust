// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use clone::Clone;
use cmp;
use container::Container;
use comm::{Sender, Receiver};
use io;
use option::{None, Option, Some};
use result::{Ok, Err};
use super::{Reader, Writer, IoResult};
use vec::{bytes, CloneableVector, MutableVector, ImmutableVector};

/// Allows reading from a rx.
///
/// # Example
///
/// ```
/// use std::io::ChanReader;
///
/// let (tx, rx) = channel();
/// # drop(tx);
/// let mut reader = ChanReader::new(rx);
///
/// let mut buf = ~[0u8, ..100];
/// match reader.read(buf) {
///     Ok(nread) => println!("Read {} bytes", nread),
///     Err(e) => println!("read error: {}", e),
/// }
/// ```
pub struct ChanReader {
    priv buf: Option<~[u8]>,  // A buffer of bytes received but not consumed.
    priv pos: uint,           // How many of the buffered bytes have already be consumed.
    priv rx: Receiver<~[u8]>,   // The rx to pull data from.
    priv closed: bool,        // Whether the pipe this rx connects to has been closed.
}

impl ChanReader {
    /// Wraps a `Port` in a `ChanReader` structure
    pub fn new(rx: Receiver<~[u8]>) -> ChanReader {
        ChanReader {
            buf: None,
            pos: 0,
            rx: rx,
            closed: false,
        }
    }
}

impl Reader for ChanReader {
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
            self.buf = self.rx.recv_opt();
            self.closed = self.buf.is_none();
        }
        if self.closed && num_read == 0 {
            Err(io::standard_error(io::EndOfFile))
        } else {
            Ok(num_read)
        }
    }
}

/// Allows writing to a tx.
///
/// # Example
///
/// ```
/// # #[allow(unused_must_use)];
/// use std::io::ChanWriter;
///
/// let (tx, rx) = channel();
/// # drop(rx);
/// let mut writer = ChanWriter::new(tx);
/// writer.write("hello, world".as_bytes());
/// ```
pub struct ChanWriter {
    priv tx: Sender<~[u8]>,
}

impl ChanWriter {
    /// Wraps a channel in a `ChanWriter` structure
    pub fn new(tx: Sender<~[u8]>) -> ChanWriter {
        ChanWriter { tx: tx }
    }
}

impl Clone for ChanWriter {
    fn clone(&self) -> ChanWriter {
        ChanWriter { tx: self.tx.clone() }
    }
}

impl Writer for ChanWriter {
    fn write(&mut self, buf: &[u8]) -> IoResult<()> {
        if !self.tx.try_send(buf.to_owned()) {
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
    fn test_rx_reader() {
        let (tx, rx) = channel();
        task::spawn(proc() {
          tx.send(~[1u8, 2u8]);
          tx.send(~[]);
          tx.send(~[3u8, 4u8]);
          tx.send(~[5u8, 6u8]);
          tx.send(~[7u8, 8u8]);
        });

        let mut reader = ChanReader::new(rx);
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
        let (tx, rx) = channel();
        let mut writer = ChanWriter::new(tx);
        writer.write_be_u32(42).unwrap();

        let wanted = ~[0u8, 0u8, 0u8, 42u8];
        let got = task::try(proc() { rx.recv() }).unwrap();
        assert_eq!(wanted, got);

        match writer.write_u8(1) {
            Ok(..) => fail!(),
            Err(e) => assert_eq!(e.kind, io::BrokenPipe),
        }
    }
}
