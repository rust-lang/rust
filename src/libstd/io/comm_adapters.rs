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

use comm::{GenericPort, GenericChan, GenericSmartChan, RecvIterator};
use io;
use super::{Reader, Writer};
use vec::CopyableVector;

/// Allows reading from a port.
///
/// # Example
///
/// ```
/// let reader = PortReader::new(port);
///
/// let mut buf = ~[0u8, ..100];
/// match reader.read(buf) {
///     Some(nread) => println!("Read {} bytes", nread),
///     None => println!("At the end of the stream!")
/// }
/// ```
pub struct PortReader<'a, P> {
    priv reader: io::extensions::BytesIterReader<RecvIterator<'a, P>>,
}

impl<'a, P: GenericPort<~[u8]>> PortReader<'a, P> {
    pub fn new(port: &'a P) -> PortReader<'a, P> {
        PortReader {
            reader: io::extensions::BytesIterReader::new(port.recv_iter()),
        }
    }
}

impl<'self, P: GenericPort<~[u8]>> Reader for PortReader<'self, P> {

    fn read(&mut self, buf: &mut [u8]) -> Option<uint> { self.reader.read(buf) }

    fn eof(&mut self) -> bool { self.reader.eof() }
}

/// Allows writing to a chan.
///
/// # Example
///
/// ```
/// let writer = ChanWriter::new(chan);
/// writer.write("hello, world".as_bytes());
/// ```
pub struct ChanWriter<C> {
    chan: C,
}

impl<C: GenericSmartChan<~[u8]>> ChanWriter<C> {
    pub fn new(chan: C) -> ChanWriter<C> {
        ChanWriter { chan: chan }
    }
}

impl<C: GenericSmartChan<~[u8]>> Writer for ChanWriter<C> {
    fn write(&mut self, buf: &[u8]) {
        if !self.chan.try_send(buf.to_owned()) {
            io::io_error::cond.raise(io::IoError {
                kind: io::BrokenPipe,
                desc: "Pipe closed",
                detail: None
            });
        }
    }
}

pub struct ReaderPort<R>;

impl<R: Reader> ReaderPort<R> {
    pub fn new(_reader: R) -> ReaderPort<R> { fail!() }
}

impl<R: Reader> GenericPort<~[u8]> for ReaderPort<R> {
    fn recv(&self) -> ~[u8] { fail!() }

    fn try_recv(&self) -> Option<~[u8]> { fail!() }
}

pub struct WriterChan<W>;

impl<W: Writer> WriterChan<W> {
    pub fn new(_writer: W) -> WriterChan<W> { fail!() }
}

impl<W: Writer> GenericChan<~[u8]> for WriterChan<W> {
    fn send(&self, _x: ~[u8]) { fail!() }
}


#[cfg(test)]
mod test {
    use prelude::*;
    use super::*;
    use io;
    use comm;
    use task;

    #[test]
    fn test_port_reader() {
        let (port, chan) = comm::stream();
        do task::spawn {
          chan.send(~[1u8, 2u8]);
          chan.send(~[]);
          chan.send(~[3u8, 4u8]);
          chan.send(~[5u8, 6u8]);
          chan.send(~[7u8, 8u8]);
        }

        let mut reader = PortReader::new(&port);
        let mut buf = ~[0u8, ..3];

        assert_eq!(false, reader.eof());

        assert_eq!(Some(0), reader.read(~[]));
        assert_eq!(false, reader.eof());

        assert_eq!(Some(3), reader.read(buf));
        assert_eq!(false, reader.eof());
        assert_eq!(~[1,2,3], buf);

        assert_eq!(Some(3), reader.read(buf));
        assert_eq!(false, reader.eof());
        assert_eq!(~[4,5,6], buf);

        assert_eq!(Some(2), reader.read(buf));
        assert_eq!(~[7,8,6], buf);
        assert_eq!(true, reader.eof());

        let mut err = None;
        let result = io::io_error::cond.trap(|io::standard_error(k, _, _)| {
            err = Some(k)
        }).inside(|| {
            reader.read(buf)
        });
        assert_eq!(Some(io::EndOfFile), err);
        assert_eq!(None, result);
        assert_eq!(true, reader.eof());
        assert_eq!(~[7,8,6], buf);

        // Ensure it continues to fail in the same way.
        err = None;
        let result = io::io_error::cond.trap(|io::standard_error(k, _, _)| {
            err = Some(k)
        }).inside(|| {
            reader.read(buf)
        });
        assert_eq!(Some(io::EndOfFile), err);
        assert_eq!(None, result);
        assert_eq!(true, reader.eof());
        assert_eq!(~[7,8,6], buf);
    }

    #[test]
    fn test_chan_writer() {
        let (port, chan) = comm::stream();
        let mut writer = ChanWriter::new(chan);
        writer.write_be_u32(42);

        let wanted = ~[0u8, 0u8, 0u8, 42u8];
        let got = do task::try { port.recv() }.unwrap();
        assert_eq!(wanted, got);

        let mut err = None;
        io::io_error::cond.trap(|io::IoError { kind, .. } | {
            err = Some(kind)
        }).inside(|| {
            writer.write_u8(1)
        });
        assert_eq!(Some(io::BrokenPipe), err);
    }
}
