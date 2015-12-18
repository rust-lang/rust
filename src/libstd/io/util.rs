// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![allow(missing_copy_implementations)]

use io::{self, Read, Write, ErrorKind, BufRead};

/// Copies the entire contents of a reader into a writer.
///
/// This function will continuously read data from `reader` and then
/// write it into `writer` in a streaming fashion until `reader`
/// returns EOF.
///
/// On success, the total number of bytes that were copied from
/// `reader` to `writer` is returned.
///
/// # Errors
///
/// This function will return an error immediately if any call to `read` or
/// `write` returns an error. All instances of `ErrorKind::Interrupted` are
/// handled by this function and the underlying operation is retried.
///
/// # Examples
///
/// ```
/// use std::io;
///
/// # fn foo() -> io::Result<()> {
/// let mut reader: &[u8] = b"hello";
/// let mut writer: Vec<u8> = vec![];
///
/// try!(io::copy(&mut reader, &mut writer));
///
/// assert_eq!(reader, &writer[..]);
/// # Ok(())
/// # }
/// ```
#[stable(feature = "rust1", since = "1.0.0")]
pub fn copy<R: ?Sized, W: ?Sized>(reader: &mut R, writer: &mut W) -> io::Result<u64>
    where R: Read, W: Write
{
    let mut buf = [0; super::DEFAULT_BUF_SIZE];
    let mut written = 0;
    loop {
        let len = match reader.read(&mut buf) {
            Ok(0) => return Ok(written),
            Ok(len) => len,
            Err(ref e) if e.kind() == ErrorKind::Interrupted => continue,
            Err(e) => return Err(e),
        };
        try!(writer.write_all(&buf[..len]));
        written += len as u64;
    }
}

/// A reader which is always at EOF.
///
/// This struct is generally created by calling [`empty()`][empty]. Please see
/// the documentation of `empty()` for more details.
///
/// [empty]: fn.empty.html
#[stable(feature = "rust1", since = "1.0.0")]
pub struct Empty { _priv: () }

/// Constructs a new handle to an empty reader.
///
/// All reads from the returned reader will return `Ok(0)`.
///
/// # Examples
///
/// A slightly sad example of not reading anything into a buffer:
///
/// ```
/// use std::io;
/// use std::io::Read;
///
/// # fn foo() -> io::Result<String> {
/// let mut buffer = String::new();
/// try!(io::empty().read_to_string(&mut buffer));
/// # Ok(buffer)
/// # }
/// ```
#[stable(feature = "rust1", since = "1.0.0")]
pub fn empty() -> Empty { Empty { _priv: () } }

#[stable(feature = "rust1", since = "1.0.0")]
impl Read for Empty {
    fn read(&mut self, _buf: &mut [u8]) -> io::Result<usize> { Ok(0) }
}
#[stable(feature = "rust1", since = "1.0.0")]
impl BufRead for Empty {
    fn fill_buf(&mut self) -> io::Result<&[u8]> { Ok(&[]) }
    fn consume(&mut self, _n: usize) {}
}

/// A reader which yields one byte over and over and over and over and over and...
///
/// This struct is generally created by calling [`repeat()`][repeat]. Please
/// see the documentation of `repeat()` for more details.
///
/// [repeat]: fn.repeat.html
#[stable(feature = "rust1", since = "1.0.0")]
pub struct Repeat { byte: u8 }

/// Creates an instance of a reader that infinitely repeats one byte.
///
/// All reads from this reader will succeed by filling the specified buffer with
/// the given byte.
#[stable(feature = "rust1", since = "1.0.0")]
pub fn repeat(byte: u8) -> Repeat { Repeat { byte: byte } }

#[stable(feature = "rust1", since = "1.0.0")]
impl Read for Repeat {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        for slot in &mut *buf {
            *slot = self.byte;
        }
        Ok(buf.len())
    }
}

/// A writer which will move data into the void.
///
/// This struct is generally created by calling [`sink()`][sink]. Please
/// see the documentation of `sink()` for more details.
///
/// [sink]: fn.sink.html
#[stable(feature = "rust1", since = "1.0.0")]
pub struct Sink { _priv: () }

/// Creates an instance of a writer which will successfully consume all data.
///
/// All calls to `write` on the returned instance will return `Ok(buf.len())`
/// and the contents of the buffer will not be inspected.
#[stable(feature = "rust1", since = "1.0.0")]
pub fn sink() -> Sink { Sink { _priv: () } }

#[stable(feature = "rust1", since = "1.0.0")]
impl Write for Sink {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> { Ok(buf.len()) }
    fn flush(&mut self) -> io::Result<()> { Ok(()) }
}

#[cfg(test)]
mod tests {
    use prelude::v1::*;

    use io::prelude::*;
    use io::{copy, sink, empty, repeat};

    #[test]
    fn copy_copies() {
        let mut r = repeat(0).take(4);
        let mut w = sink();
        assert_eq!(copy(&mut r, &mut w).unwrap(), 4);

        let mut r = repeat(0).take(1 << 17);
        assert_eq!(copy(&mut r as &mut Read, &mut w as &mut Write).unwrap(), 1 << 17);
    }

    #[test]
    fn sink_sinks() {
        let mut s = sink();
        assert_eq!(s.write(&[]).unwrap(), 0);
        assert_eq!(s.write(&[0]).unwrap(), 1);
        assert_eq!(s.write(&[0; 1024]).unwrap(), 1024);
        assert_eq!(s.by_ref().write(&[0; 1024]).unwrap(), 1024);
    }

    #[test]
    fn empty_reads() {
        let mut e = empty();
        assert_eq!(e.read(&mut []).unwrap(), 0);
        assert_eq!(e.read(&mut [0]).unwrap(), 0);
        assert_eq!(e.read(&mut [0; 1024]).unwrap(), 0);
        assert_eq!(e.by_ref().read(&mut [0; 1024]).unwrap(), 0);
    }

    #[test]
    fn repeat_repeats() {
        let mut r = repeat(4);
        let mut b = [0; 1024];
        assert_eq!(r.read(&mut b).unwrap(), 1024);
        assert!(b.iter().all(|b| *b == 4));
    }

    #[test]
    fn take_some_bytes() {
        assert_eq!(repeat(4).take(100).bytes().count(), 100);
        assert_eq!(repeat(4).take(100).bytes().next().unwrap().unwrap(), 4);
        assert_eq!(repeat(1).take(10).chain(repeat(2).take(10)).bytes().count(), 20);
    }

    #[test]
    #[allow(deprecated)]
    fn tee() {
        let mut buf = [0; 10];
        {
            let mut ptr: &mut [u8] = &mut buf;
            assert_eq!(repeat(4).tee(&mut ptr).take(5).read(&mut [0; 10]).unwrap(), 5);
        }
        assert_eq!(buf, [4, 4, 4, 4, 4, 0, 0, 0, 0, 0]);
    }

    #[test]
    #[allow(deprecated)]
    fn broadcast() {
        let mut buf1 = [0; 10];
        let mut buf2 = [0; 10];
        {
            let mut ptr1: &mut [u8] = &mut buf1;
            let mut ptr2: &mut [u8] = &mut buf2;

            assert_eq!((&mut ptr1).broadcast(&mut ptr2)
                                  .write(&[1, 2, 3]).unwrap(), 3);
        }
        assert_eq!(buf1, buf2);
        assert_eq!(buf1, [1, 2, 3, 0, 0, 0, 0, 0, 0, 0]);
    }
}
