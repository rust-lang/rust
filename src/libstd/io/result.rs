// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Implementations of I/O traits for the IoResult type
//!
//! I/O constructors return option types to allow errors to be handled.
//! These implementations allow e.g. `IoResult<File>` to be used
//! as a `Reader` without unwrapping the result first.

use clone::Clone;
use result::{Ok, Err};
use super::{Reader, Writer, Listener, Acceptor, Seek, SeekStyle, IoResult};

impl<W: Writer> Writer for IoResult<W> {
    fn write(&mut self, buf: &[u8]) -> IoResult<()> {
        match *self {
            Ok(ref mut writer) => writer.write(buf),
            Err(ref e) => Err((*e).clone())
        }
    }

    fn flush(&mut self) -> IoResult<()> {
        match *self {
            Ok(ref mut writer) => writer.flush(),
            Err(ref e) => Err(e.clone()),
        }
    }
}

impl<R: Reader> Reader for IoResult<R> {
    fn read(&mut self, buf: &mut [u8]) -> IoResult<uint> {
        match *self {
            Ok(ref mut reader) => reader.read(buf),
            Err(ref e) => Err(e.clone()),
        }
    }
}

impl<S: Seek> Seek for IoResult<S> {
    fn tell(&self) -> IoResult<u64> {
        match *self {
            Ok(ref seeker) => seeker.tell(),
            Err(ref e) => Err(e.clone()),
        }
    }
    fn seek(&mut self, pos: i64, style: SeekStyle) -> IoResult<()> {
        match *self {
            Ok(ref mut seeker) => seeker.seek(pos, style),
            Err(ref e) => Err(e.clone())
        }
    }
}

impl<T, A: Acceptor<T>, L: Listener<T, A>> Listener<T, A> for IoResult<L> {
    fn listen(self) -> IoResult<A> {
        match self {
            Ok(listener) => listener.listen(),
            Err(e) => Err(e),
        }
    }
}

impl<T, A: Acceptor<T>> Acceptor<T> for IoResult<A> {
    fn accept(&mut self) -> IoResult<T> {
        match *self {
            Ok(ref mut acceptor) => acceptor.accept(),
            Err(ref e) => Err(e.clone()),
        }
    }
}

#[cfg(test)]
mod test {
    use prelude::*;
    use super::super::mem::*;
    use io;

    #[test]
    fn test_option_writer() {
        let mut writer: io::IoResult<MemWriter> = Ok(MemWriter::new());
        writer.write([0, 1, 2]).unwrap();
        writer.flush().unwrap();
        assert_eq!(writer.unwrap().unwrap(), vec!(0, 1, 2));
    }

    #[test]
    fn test_option_writer_error() {
        let mut writer: io::IoResult<MemWriter> =
            Err(io::standard_error(io::EndOfFile));

        match writer.write([0, 0, 0]) {
            Ok(..) => fail!(),
            Err(e) => assert_eq!(e.kind, io::EndOfFile),
        }
        match writer.flush() {
            Ok(..) => fail!(),
            Err(e) => assert_eq!(e.kind, io::EndOfFile),
        }
    }

    #[test]
    fn test_option_reader() {
        let mut reader: io::IoResult<MemReader> =
            Ok(MemReader::new(vec!(0, 1, 2, 3)));
        let mut buf = [0, 0];
        reader.read(buf).unwrap();
        let b: &[_] = &[0, 1];
        assert_eq!(buf.as_slice(), b);
    }

    #[test]
    fn test_option_reader_error() {
        let mut reader: io::IoResult<MemReader> =
            Err(io::standard_error(io::EndOfFile));
        let mut buf = [];

        match reader.read(buf) {
            Ok(..) => fail!(),
            Err(e) => assert_eq!(e.kind, io::EndOfFile),
        }
    }
}
