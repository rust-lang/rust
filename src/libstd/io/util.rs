#![allow(missing_copy_implementations)]

use crate::fmt;
use crate::io::{self, Read, Initializer, Write, ErrorKind, BufRead, IoSlice, IoSliceMut};
use crate::mem;

/// Copies the entire contents of a reader into a writer.
///
/// This function will continuously read data from `reader` and then
/// write it into `writer` in a streaming fashion until `reader`
/// returns EOF.
///
/// On success, the total number of bytes that were copied from
/// `reader` to `writer` is returned.
///
/// If you’re wanting to copy the contents of one file to another and you’re
/// working with filesystem paths, see the [`fs::copy`] function.
///
/// [`fs::copy`]: ../fs/fn.copy.html
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
/// fn main() -> io::Result<()> {
///     let mut reader: &[u8] = b"hello";
///     let mut writer: Vec<u8> = vec![];
///
///     io::copy(&mut reader, &mut writer)?;
///
///     assert_eq!(&b"hello"[..], &writer[..]);
///     Ok(())
/// }
/// ```
#[stable(feature = "rust1", since = "1.0.0")]
pub fn copy<R: ?Sized, W: ?Sized>(reader: &mut R, writer: &mut W) -> io::Result<u64>
    where R: Read, W: Write
{
    let mut buf = unsafe {
        #[allow(deprecated)]
        let mut buf: [u8; super::DEFAULT_BUF_SIZE] = mem::uninitialized();
        reader.initializer().initialize(&mut buf);
        buf
    };

    let mut written = 0;
    loop {
        let len = match reader.read(&mut buf) {
            Ok(0) => return Ok(written),
            Ok(len) => len,
            Err(ref e) if e.kind() == ErrorKind::Interrupted => continue,
            Err(e) => return Err(e),
        };
        writer.write_all(&buf[..len])?;
        written += len as u64;
    }
}

/// A reader which is always at EOF.
///
/// This struct is generally created by calling [`empty`]. Please see
/// the documentation of [`empty()`][`empty`] for more details.
///
/// [`empty`]: fn.empty.html
#[stable(feature = "rust1", since = "1.0.0")]
pub struct Empty { _priv: () }

/// Constructs a new handle to an empty reader.
///
/// All reads from the returned reader will return [`Ok`]`(0)`.
///
/// [`Ok`]: ../result/enum.Result.html#variant.Ok
///
/// # Examples
///
/// A slightly sad example of not reading anything into a buffer:
///
/// ```
/// use std::io::{self, Read};
///
/// let mut buffer = String::new();
/// io::empty().read_to_string(&mut buffer).unwrap();
/// assert!(buffer.is_empty());
/// ```
#[stable(feature = "rust1", since = "1.0.0")]
pub fn empty() -> Empty { Empty { _priv: () } }

#[stable(feature = "rust1", since = "1.0.0")]
impl Read for Empty {
    #[inline]
    fn read(&mut self, _buf: &mut [u8]) -> io::Result<usize> { Ok(0) }

    #[inline]
    unsafe fn initializer(&self) -> Initializer {
        Initializer::nop()
    }
}
#[stable(feature = "rust1", since = "1.0.0")]
impl BufRead for Empty {
    #[inline]
    fn fill_buf(&mut self) -> io::Result<&[u8]> { Ok(&[]) }
    #[inline]
    fn consume(&mut self, _n: usize) {}
}

#[stable(feature = "std_debug", since = "1.16.0")]
impl fmt::Debug for Empty {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.pad("Empty { .. }")
    }
}

/// A reader which yields one byte over and over and over and over and over and...
///
/// This struct is generally created by calling [`repeat`][repeat]. Please
/// see the documentation of `repeat()` for more details.
///
/// [repeat]: fn.repeat.html
#[stable(feature = "rust1", since = "1.0.0")]
pub struct Repeat { byte: u8 }

/// Creates an instance of a reader that infinitely repeats one byte.
///
/// All reads from this reader will succeed by filling the specified buffer with
/// the given byte.
///
/// # Examples
///
/// ```
/// use std::io::{self, Read};
///
/// let mut buffer = [0; 3];
/// io::repeat(0b101).read_exact(&mut buffer).unwrap();
/// assert_eq!(buffer, [0b101, 0b101, 0b101]);
/// ```
#[stable(feature = "rust1", since = "1.0.0")]
pub fn repeat(byte: u8) -> Repeat { Repeat { byte } }

#[stable(feature = "rust1", since = "1.0.0")]
impl Read for Repeat {
    #[inline]
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        for slot in &mut *buf {
            *slot = self.byte;
        }
        Ok(buf.len())
    }

    #[inline]
    fn read_vectored(&mut self, bufs: &mut [IoSliceMut<'_>]) -> io::Result<usize> {
        let mut nwritten = 0;
        for buf in bufs {
            nwritten += self.read(buf)?;
        }
        Ok(nwritten)
    }

    #[inline]
    unsafe fn initializer(&self) -> Initializer {
        Initializer::nop()
    }
}

#[stable(feature = "std_debug", since = "1.16.0")]
impl fmt::Debug for Repeat {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.pad("Repeat { .. }")
    }
}

/// A writer which will move data into the void.
///
/// This struct is generally created by calling [`sink`][sink]. Please
/// see the documentation of `sink()` for more details.
///
/// [sink]: fn.sink.html
#[stable(feature = "rust1", since = "1.0.0")]
pub struct Sink { _priv: () }

/// Creates an instance of a writer which will successfully consume all data.
///
/// All calls to `write` on the returned instance will return `Ok(buf.len())`
/// and the contents of the buffer will not be inspected.
///
/// # Examples
///
/// ```rust
/// use std::io::{self, Write};
///
/// let buffer = vec![1, 2, 3, 5, 8];
/// let num_bytes = io::sink().write(&buffer).unwrap();
/// assert_eq!(num_bytes, 5);
/// ```
#[stable(feature = "rust1", since = "1.0.0")]
pub fn sink() -> Sink { Sink { _priv: () } }

#[stable(feature = "rust1", since = "1.0.0")]
impl Write for Sink {
    #[inline]
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> { Ok(buf.len()) }

    #[inline]
    fn write_vectored(&mut self, bufs: &[IoSlice<'_>]) -> io::Result<usize> {
        let total_len = bufs.iter().map(|b| b.len()).sum();
        Ok(total_len)
    }

    #[inline]
    fn flush(&mut self) -> io::Result<()> { Ok(()) }
}

#[stable(feature = "std_debug", since = "1.16.0")]
impl fmt::Debug for Sink {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.pad("Sink { .. }")
    }
}

#[cfg(test)]
mod tests {
    use crate::io::prelude::*;
    use crate::io::{copy, sink, empty, repeat};

    #[test]
    fn copy_copies() {
        let mut r = repeat(0).take(4);
        let mut w = sink();
        assert_eq!(copy(&mut r, &mut w).unwrap(), 4);

        let mut r = repeat(0).take(1 << 17);
        assert_eq!(copy(&mut r as &mut dyn Read, &mut w as &mut dyn Write).unwrap(), 1 << 17);
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
}
