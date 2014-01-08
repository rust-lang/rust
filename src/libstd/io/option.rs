// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Implementations of I/O traits for the Option type
//!
//! I/O constructors return option types to allow errors to be handled.
//! These implementations allow e.g. `Option<File>` to be used
//! as a `Reader` without unwrapping the option first.

use option::*;
use super::{Reader, Writer, Listener, Acceptor, Seek, SeekStyle};
use super::{standard_error, PreviousIoError, io_error, IoError};

fn prev_io_error() -> IoError {
    standard_error(PreviousIoError)
}

impl<W: Writer> Writer for Option<W> {
    fn write(&mut self, buf: &[u8]) {
        match *self {
            Some(ref mut writer) => writer.write(buf),
            None => io_error::cond.raise(prev_io_error())
        }
    }

    fn flush(&mut self) {
        match *self {
            Some(ref mut writer) => writer.flush(),
            None => io_error::cond.raise(prev_io_error())
        }
    }
}

impl<R: Reader> Reader for Option<R> {
    fn read(&mut self, buf: &mut [u8]) -> Option<uint> {
        match *self {
            Some(ref mut reader) => reader.read(buf),
            None => {
                io_error::cond.raise(prev_io_error());
                None
            }
        }
    }

    fn eof(&mut self) -> bool {
        match *self {
            Some(ref mut reader) => reader.eof(),
            None => {
                io_error::cond.raise(prev_io_error());
                true
            }
        }
    }
}

impl<S: Seek> Seek for Option<S> {
    fn tell(&self) -> u64 {
        match *self {
            Some(ref seeker) => seeker.tell(),
            None => {
                io_error::cond.raise(prev_io_error());
                0
            }
        }
    }
    fn seek(&mut self, pos: i64, style: SeekStyle) {
        match *self {
            Some(ref mut seeker) => seeker.seek(pos, style),
            None => io_error::cond.raise(prev_io_error())
        }
    }
}

impl<T, A: Acceptor<T>, L: Listener<T, A>> Listener<T, A> for Option<L> {
    fn listen(self) -> Option<A> {
        match self {
            Some(listener) => listener.listen(),
            None => {
                io_error::cond.raise(prev_io_error());
                None
            }
        }
    }
}

impl<T, A: Acceptor<T>> Acceptor<T> for Option<A> {
    fn accept(&mut self) -> Option<T> {
        match *self {
            Some(ref mut acceptor) => acceptor.accept(),
            None => {
                io_error::cond.raise(prev_io_error());
                None
            }
        }
    }
}

#[cfg(test)]
mod test {
    use prelude::*;
    use super::super::mem::*;
    use io::Decorator;
    use super::super::{PreviousIoError, io_error};

    #[test]
    fn test_option_writer() {
        let mut writer: Option<MemWriter> = Some(MemWriter::new());
        writer.write([0, 1, 2]);
        writer.flush();
        assert_eq!(writer.unwrap().inner(), ~[0, 1, 2]);
    }

    #[test]
    fn test_option_writer_error() {
        let mut writer: Option<MemWriter> = None;

        let mut called = false;
        io_error::cond.trap(|err| {
            assert_eq!(err.kind, PreviousIoError);
            called = true;
        }).inside(|| {
            writer.write([0, 0, 0]);
        });
        assert!(called);

        let mut called = false;
        io_error::cond.trap(|err| {
            assert_eq!(err.kind, PreviousIoError);
            called = true;
        }).inside(|| {
            writer.flush();
        });
        assert!(called);
    }

    #[test]
    fn test_option_reader() {
        let mut reader: Option<MemReader> = Some(MemReader::new(~[0, 1, 2, 3]));
        let mut buf = [0, 0];
        reader.read(buf);
        assert_eq!(buf, [0, 1]);
        assert!(!reader.eof());
    }

    #[test]
    fn test_option_reader_error() {
        let mut reader: Option<MemReader> = None;
        let mut buf = [];

        let mut called = false;
        io_error::cond.trap(|err| {
            assert_eq!(err.kind, PreviousIoError);
            called = true;
        }).inside(|| {
            reader.read(buf);
        });
        assert!(called);

        let mut called = false;
        io_error::cond.trap(|err| {
            assert_eq!(err.kind, PreviousIoError);
            called = true;
        }).inside(|| {
            assert!(reader.eof());
        });
        assert!(called);
    }
}
