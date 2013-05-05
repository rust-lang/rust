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
//! These implementations allow e.g. `Option<FileStream>` to be used
//! as a `Reader` without unwrapping the option first.
//!
//! # XXX Seek and Close

use option::*;
use super::{Reader, Writer, Listener};
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

impl<L: Listener<S>, S> Listener<S> for Option<L> {
    fn accept(&mut self) -> Option<S> {
        match *self {
            Some(ref mut listener) => listener.accept(),
            None => {
                io_error::cond.raise(prev_io_error());
                None
            }
        }
    }
}

#[cfg(test)]
mod test {
    use option::*;
    use super::super::mem::*;
    use rt::test::*;
    use super::super::{PreviousIoError, io_error};

    #[test]
    fn test_option_writer() {
        do run_in_newsched_task {
            let mut writer: Option<MemWriter> = Some(MemWriter::new());
            writer.write([0, 1, 2]);
            writer.flush();
            assert!(writer.unwrap().inner() == ~[0, 1, 2]);
        }
    }

    #[test]
    fn test_option_writer_error() {
        do run_in_newsched_task {
            let mut writer: Option<MemWriter> = None;

            let mut called = false;
            do io_error::cond.trap(|err| {
                assert!(err.kind == PreviousIoError);
                called = true;
            }).in {
                writer.write([0, 0, 0]);
            }
            assert!(called);

            let mut called = false;
            do io_error::cond.trap(|err| {
                assert!(err.kind == PreviousIoError);
                called = true;
            }).in {
                writer.flush();
            }
            assert!(called);
        }
    }

    #[test]
    fn test_option_reader() {
        do run_in_newsched_task {
            let mut reader: Option<MemReader> = Some(MemReader::new(~[0, 1, 2, 3]));
            let mut buf = [0, 0];
            reader.read(buf);
            assert!(buf == [0, 1]);
            assert!(!reader.eof());
        }
    }

    #[test]
    fn test_option_reader_error() {
        let mut reader: Option<MemReader> = None;
        let mut buf = [];

        let mut called = false;
        do io_error::cond.trap(|err| {
            assert!(err.kind == PreviousIoError);
            called = true;
        }).in {
            reader.read(buf);
        }
        assert!(called);

        let mut called = false;
        do io_error::cond.trap(|err| {
            assert!(err.kind == PreviousIoError);
            called = true;
        }).in {
            assert!(reader.eof());
        }
        assert!(called);
    }
}
