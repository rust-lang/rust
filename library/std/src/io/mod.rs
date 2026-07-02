//! Traits, helpers, and type definitions for core I/O functionality.
//!
//! The `std::io` module contains a number of common things you'll need
//! when doing input and output. The most core part of this module is
//! the [`Read`] and [`Write`] traits, which provide the
//! most general interface for reading and writing input and output.
//!
//! ## Read and Write
//!
//! Because they are traits, [`Read`] and [`Write`] are implemented by a number
//! of other types, and you can implement them for your types too. As such,
//! you'll see a few different types of I/O throughout the documentation in
//! this module: [`File`]s, [`TcpStream`]s, and sometimes even [`Vec<T>`]s. For
//! example, [`Read`] adds a [`read`][`Read::read`] method, which we can use on
//! [`File`]s:
//!
//! ```no_run
//! use std::io;
//! use std::io::prelude::*;
//! use std::fs::File;
//!
//! fn main() -> io::Result<()> {
//!     let mut f = File::open("foo.txt")?;
//!     let mut buffer = [0; 10];
//!
//!     // read up to 10 bytes
//!     let n = f.read(&mut buffer)?;
//!
//!     println!("The bytes: {:?}", &buffer[..n]);
//!     Ok(())
//! }
//! ```
//!
//! [`Read`] and [`Write`] are so important, implementors of the two traits have a
//! nickname: readers and writers. So you'll sometimes see 'a reader' instead
//! of 'a type that implements the [`Read`] trait'. Much easier!
//!
//! ## Seek and BufRead
//!
//! Beyond that, there are two important traits that are provided: [`Seek`]
//! and [`BufRead`]. Both of these build on top of a reader to control
//! how the reading happens. [`Seek`] lets you control where the next byte is
//! coming from:
//!
//! ```no_run
//! use std::io;
//! use std::io::prelude::*;
//! use std::io::SeekFrom;
//! use std::fs::File;
//!
//! fn main() -> io::Result<()> {
//!     let mut f = File::open("foo.txt")?;
//!     let mut buffer = [0; 10];
//!
//!     // skip to the last 10 bytes of the file
//!     f.seek(SeekFrom::End(-10))?;
//!
//!     // read up to 10 bytes
//!     let n = f.read(&mut buffer)?;
//!
//!     println!("The bytes: {:?}", &buffer[..n]);
//!     Ok(())
//! }
//! ```
//!
//! [`BufRead`] uses an internal buffer to provide a number of other ways to read, but
//! to show it off, we'll need to talk about buffers in general. Keep reading!
//!
//! ## BufReader and BufWriter
//!
//! Byte-based interfaces are unwieldy and can be inefficient, as we'd need to be
//! making near-constant calls to the operating system. To help with this,
//! `std::io` comes with two structs, [`BufReader`] and [`BufWriter`], which wrap
//! readers and writers. The wrapper uses a buffer, reducing the number of
//! calls and providing nicer methods for accessing exactly what you want.
//!
//! For example, [`BufReader`] works with the [`BufRead`] trait to add extra
//! methods to any reader:
//!
//! ```no_run
//! use std::io;
//! use std::io::prelude::*;
//! use std::io::BufReader;
//! use std::fs::File;
//!
//! fn main() -> io::Result<()> {
//!     let f = File::open("foo.txt")?;
//!     let mut reader = BufReader::new(f);
//!     let mut buffer = String::new();
//!
//!     // read a line into buffer
//!     reader.read_line(&mut buffer)?;
//!
//!     println!("{buffer}");
//!     Ok(())
//! }
//! ```
//!
//! [`BufWriter`] doesn't add any new ways of writing; it just buffers every call
//! to [`write`][`Write::write`]:
//!
//! ```no_run
//! use std::io;
//! use std::io::prelude::*;
//! use std::io::BufWriter;
//! use std::fs::File;
//!
//! fn main() -> io::Result<()> {
//!     let f = File::create("foo.txt")?;
//!     {
//!         let mut writer = BufWriter::new(f);
//!
//!         // write a byte to the buffer
//!         writer.write(&[42])?;
//!
//!     } // the buffer is flushed once writer goes out of scope
//!
//!     Ok(())
//! }
//! ```
//!
//! ## Standard input and output
//!
//! A very common source of input is standard input:
//!
//! ```no_run
//! use std::io;
//!
//! fn main() -> io::Result<()> {
//!     let mut input = String::new();
//!
//!     io::stdin().read_line(&mut input)?;
//!
//!     println!("You typed: {}", input.trim());
//!     Ok(())
//! }
//! ```
//!
//! Note that you cannot use the [`?` operator] in functions that do not return
//! a [`Result<T, E>`][`Result`]. Instead, you can call [`.unwrap()`]
//! or `match` on the return value to catch any possible errors:
//!
//! ```no_run
//! use std::io;
//!
//! let mut input = String::new();
//!
//! io::stdin().read_line(&mut input).unwrap();
//! ```
//!
//! And a very common source of output is standard output:
//!
//! ```no_run
//! use std::io;
//! use std::io::prelude::*;
//!
//! fn main() -> io::Result<()> {
//!     io::stdout().write(&[42])?;
//!     Ok(())
//! }
//! ```
//!
//! Of course, using [`io::stdout`] directly is less common than something like
//! [`println!`].
//!
//! ## Iterator types
//!
//! A large number of the structures provided by `std::io` are for various
//! ways of iterating over I/O. For example, [`Lines`] is used to split over
//! lines:
//!
//! ```no_run
//! use std::io;
//! use std::io::prelude::*;
//! use std::io::BufReader;
//! use std::fs::File;
//!
//! fn main() -> io::Result<()> {
//!     let f = File::open("foo.txt")?;
//!     let reader = BufReader::new(f);
//!
//!     for line in reader.lines() {
//!         println!("{}", line?);
//!     }
//!     Ok(())
//! }
//! ```
//!
//! ## Functions
//!
//! There are a number of [functions][functions-list] that offer access to various
//! features. For example, we can use three of these functions to copy everything
//! from standard input to standard output:
//!
//! ```no_run
//! use std::io;
//!
//! fn main() -> io::Result<()> {
//!     io::copy(&mut io::stdin(), &mut io::stdout())?;
//!     Ok(())
//! }
//! ```
//!
//! [functions-list]: #functions-1
//!
//! ## io::Result
//!
//! Last, but certainly not least, is [`io::Result`]. This type is used
//! as the return type of many `std::io` functions that can cause an error, and
//! can be returned from your own functions as well. Many of the examples in this
//! module use the [`?` operator]:
//!
//! ```
//! use std::io;
//!
//! fn read_input() -> io::Result<()> {
//!     let mut input = String::new();
//!
//!     io::stdin().read_line(&mut input)?;
//!
//!     println!("You typed: {}", input.trim());
//!
//!     Ok(())
//! }
//! ```
//!
//! The return type of `read_input()`, [`io::Result<()>`][`io::Result`], is a very
//! common type for functions which don't have a 'real' return value, but do want to
//! return errors if they happen. In this case, the only purpose of this function is
//! to read the line and print it, so we use `()`.
//!
//! ## Platform-specific behavior
//!
//! Many I/O functions throughout the standard library are documented to indicate
//! what various library or syscalls they are delegated to. This is done to help
//! applications both understand what's happening under the hood as well as investigate
//! any possibly unclear semantics. Note, however, that this is informative, not a binding
//! contract. The implementation of many of these functions are subject to change over
//! time and may call fewer or more syscalls/library functions.
//!
//! ## I/O Safety
//!
//! Rust follows an I/O safety discipline that is comparable to its memory safety discipline. This
//! means that file descriptors can be *exclusively owned*. (Here, "file descriptor" is meant to
//! subsume similar concepts that exist across a wide range of operating systems even if they might
//! use a different name, such as "handle".) An exclusively owned file descriptor is one that no
//! other code is allowed to access in any way, but the owner is allowed to access and even close
//! it any time. A type that owns its file descriptor should usually close it in its `drop`
//! function. Types like [`File`] own their file descriptor. Similarly, file descriptors
//! can be *borrowed*, granting the temporary right to perform operations on this file descriptor.
//! This indicates that the file descriptor will not be closed for the lifetime of the borrow, but
//! it does *not* imply any right to close this file descriptor, since it will likely be owned by
//! someone else.
//!
//! The platform-specific parts of the Rust standard library expose types that reflect these
//! concepts, see [`os::unix`] and [`os::windows`].
//!
//! To uphold I/O safety, it is crucial that no code acts on file descriptors it does not own or
//! borrow, and no code closes file descriptors it does not own. In other words, a safe function
//! that takes a regular integer, treats it as a file descriptor, and acts on it, is *unsound*.
//!
//! Not upholding I/O safety and acting on a file descriptor without proof of ownership can lead to
//! misbehavior and even Undefined Behavior in code that relies on ownership of its file
//! descriptors: a closed file descriptor could be re-allocated, so the original owner of that file
//! descriptor is now working on the wrong file. Some code might even rely on fully encapsulating
//! its file descriptors with no operations being performed by any other part of the program.
//!
//! Note that exclusive ownership of a file descriptor does *not* imply exclusive ownership of the
//! underlying kernel object that the file descriptor references (also called "open file description" on
//! some operating systems). File descriptors basically work like [`Arc`]: when you receive an owned
//! file descriptor, you cannot know whether there are any other file descriptors that reference the
//! same kernel object. However, when you create a new kernel object, you know that you are holding
//! the only reference to it. Just be careful not to lend it to anyone, since they can obtain a
//! clone and then you can no longer know what the reference count is! In that sense, [`OwnedFd`] is
//! like `Arc` and [`BorrowedFd<'a>`] is like `&'a Arc` (and similar for the Windows types). In
//! particular, given a `BorrowedFd<'a>`, you are not allowed to close the file descriptor -- just
//! like how, given a `&'a Arc`, you are not allowed to decrement the reference count and
//! potentially free the underlying object. There is no equivalent to `Box` for file descriptors in
//! the standard library (that would be a type that guarantees that the reference count is `1`),
//! however, it would be possible for a crate to define a type with those semantics.
//!
//! [`File`]: crate::fs::File
//! [`TcpStream`]: crate::net::TcpStream
//! [`io::stdout`]: stdout
//! [`io::Result`]: self::Result
//! [`?` operator]: ../../book/appendix-02-operators.html
//! [`Result`]: crate::result::Result
//! [`.unwrap()`]: crate::result::Result::unwrap
//! [`os::unix`]: ../os/unix/io/index.html
//! [`os::windows`]: ../os/windows/io/index.html
//! [`OwnedFd`]: ../os/fd/struct.OwnedFd.html
//! [`BorrowedFd<'a>`]: ../os/fd/struct.BorrowedFd.html
//! [`Arc`]: crate::sync::Arc

#![stable(feature = "rust1", since = "1.0.0")]

#[cfg(test)]
mod tests;

use core::slice::memchr;

#[unstable(feature = "raw_os_error_ty", issue = "107792")]
pub use alloc_crate::io::RawOsError;
#[doc(hidden)]
#[unstable(feature = "io_const_error_internals", issue = "none")]
pub use alloc_crate::io::SimpleMessage;
#[unstable(feature = "io_const_error", issue = "133448")]
pub use alloc_crate::io::const_error;
#[stable(feature = "io_read_to_string", since = "1.65.0")]
pub use alloc_crate::io::read_to_string;
#[unstable(feature = "read_buf", issue = "78485")]
pub use alloc_crate::io::{BorrowedBuf, BorrowedCursor};
#[stable(feature = "rust1", since = "1.0.0")]
pub use alloc_crate::io::{
    Bytes, Chain, Empty, Error, ErrorKind, Read, Repeat, Result, Seek, SeekFrom, Sink, Take, Write,
    empty, repeat, sink,
};
#[allow(unused_imports, reason = "only used by certain target configurations")]
pub(crate) use alloc_crate::io::{DEFAULT_BUF_SIZE, default_read_buf};
pub(crate) use alloc_crate::io::{
    IoHandle, SpecReadByte, append_to_string, default_read_buf_exact, default_read_exact,
    default_read_to_end, default_read_to_string, default_read_vectored, default_write_vectored,
    stream_len_default, uninlined_slow_read_byte,
};
#[stable(feature = "iovec", since = "1.36.0")]
pub use alloc_crate::io::{IoSlice, IoSliceMut};
use alloc_crate::io::{OsFunctions, SizeHint};

#[stable(feature = "bufwriter_into_parts", since = "1.56.0")]
pub use self::buffered::WriterPanicked;
#[stable(feature = "anonymous_pipe", since = "1.87.0")]
pub use self::pipe::{PipeReader, PipeWriter, pipe};
#[stable(feature = "is_terminal", since = "1.70.0")]
pub use self::stdio::IsTerminal;
pub(crate) use self::stdio::attempt_print_to_stderr;
#[unstable(feature = "print_internals", issue = "none")]
#[doc(hidden)]
pub use self::stdio::{_eprint, _print};
#[unstable(feature = "internal_output_capture", issue = "none")]
#[doc(no_inline, hidden)]
pub use self::stdio::{set_output_capture, try_set_output_capture};
#[stable(feature = "rust1", since = "1.0.0")]
pub use self::{
    buffered::{BufReader, BufWriter, IntoInnerError, LineWriter},
    copy::copy,
    cursor::Cursor,
    stdio::{Stderr, StderrLock, Stdin, StdinLock, Stdout, StdoutLock, stderr, stdin, stdout},
};
use crate::cmp;

mod buffered;
pub(crate) mod copy;
mod cursor;
mod error;
mod impls;
mod pipe;
pub mod prelude;
mod stdio;
mod util;

pub(crate) use stdio::cleanup;

fn read_until<R: BufRead + ?Sized>(r: &mut R, delim: u8, buf: &mut Vec<u8>) -> Result<usize> {
    let mut read = 0;
    loop {
        let (done, used) = {
            let available = match r.fill_buf() {
                Ok(n) => n,
                Err(ref e) if e.is_interrupted() => continue,
                Err(e) => return Err(e),
            };
            match memchr::memchr(delim, available) {
                Some(i) => {
                    buf.extend_from_slice(&available[..=i]);
                    (true, i + 1)
                }
                None => {
                    buf.extend_from_slice(available);
                    (false, available.len())
                }
            }
        };
        r.consume(used);
        read += used;
        if done || used == 0 {
            return Ok(read);
        }
    }
}

fn skip_until<R: BufRead + ?Sized>(r: &mut R, delim: u8) -> Result<usize> {
    let mut read = 0;
    loop {
        let (done, used) = {
            let available = match r.fill_buf() {
                Ok(n) => n,
                Err(ref e) if e.kind() == ErrorKind::Interrupted => continue,
                Err(e) => return Err(e),
            };
            match memchr::memchr(delim, available) {
                Some(i) => (true, i + 1),
                None => (false, available.len()),
            }
        };
        r.consume(used);
        read += used;
        if done || used == 0 {
            return Ok(read);
        }
    }
}

/// A `BufRead` is a type of `Read`er which has an internal buffer, allowing it
/// to perform extra ways of reading.
///
/// For example, reading line-by-line is inefficient without using a buffer, so
/// if you want to read by line, you'll need `BufRead`, which includes a
/// [`read_line`] method as well as a [`lines`] iterator.
///
/// # Examples
///
/// A locked standard input implements `BufRead`:
///
/// ```no_run
/// use std::io;
/// use std::io::prelude::*;
///
/// let stdin = io::stdin();
/// for line in stdin.lock().lines() {
///     println!("{}", line?);
/// }
/// # std::io::Result::Ok(())
/// ```
///
/// If you have something that implements [`Read`], you can use the [`BufReader`
/// type][`BufReader`] to turn it into a `BufRead`.
///
/// For example, [`File`] implements [`Read`], but not `BufRead`.
/// [`BufReader`] to the rescue!
///
/// [`File`]: crate::fs::File
/// [`read_line`]: BufRead::read_line
/// [`lines`]: BufRead::lines
///
/// ```no_run
/// use std::io::{self, BufReader};
/// use std::io::prelude::*;
/// use std::fs::File;
///
/// fn main() -> io::Result<()> {
///     let f = File::open("foo.txt")?;
///     let f = BufReader::new(f);
///
///     for line in f.lines() {
///         let line = line?;
///         println!("{line}");
///     }
///
///     Ok(())
/// }
/// ```
#[stable(feature = "rust1", since = "1.0.0")]
#[cfg_attr(not(test), rustc_diagnostic_item = "IoBufRead")]
pub trait BufRead: Read {
    /// Returns the contents of the internal buffer, filling it with more data, via `Read` methods, if empty.
    ///
    /// This is a lower-level method and is meant to be used together with [`consume`],
    /// which can be used to mark bytes that should not be returned by subsequent calls to `read`.
    ///
    /// [`consume`]: BufRead::consume
    ///
    /// Returns an empty buffer when the stream has reached EOF.
    ///
    /// # Errors
    ///
    /// This function will return an I/O error if a `Read` method was called, but returned an error.
    ///
    /// # Examples
    ///
    /// A locked standard input implements `BufRead`:
    ///
    /// ```no_run
    /// use std::io;
    /// use std::io::prelude::*;
    ///
    /// let stdin = io::stdin();
    /// let mut stdin = stdin.lock();
    ///
    /// let buffer = stdin.fill_buf()?;
    ///
    /// // work with buffer
    /// println!("{buffer:?}");
    ///
    /// // mark the bytes we worked with as read
    /// let length = buffer.len();
    /// stdin.consume(length);
    /// # std::io::Result::Ok(())
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    fn fill_buf(&mut self) -> Result<&[u8]>;

    /// Marks the given `amount` of additional bytes from the internal buffer as having been read.
    /// Subsequent calls to `read` only return bytes that have not been marked as read.
    ///
    /// This is a lower-level method and is meant to be used together with [`fill_buf`],
    /// which can be used to fill the internal buffer via `Read` methods.
    ///
    /// It is a logic error if `amount` exceeds the number of unread bytes in the internal buffer, which is returned by [`fill_buf`].
    ///
    /// # Examples
    ///
    /// Since `consume()` is meant to be used with [`fill_buf`],
    /// that method's example includes an example of `consume()`.
    ///
    /// [`fill_buf`]: BufRead::fill_buf
    #[stable(feature = "rust1", since = "1.0.0")]
    fn consume(&mut self, amount: usize);

    /// Checks if there is any data left to be `read`.
    ///
    /// This function may fill the buffer to check for data,
    /// so this function returns `Result<bool>`, not `bool`.
    ///
    /// The default implementation calls `fill_buf` and checks that the
    /// returned slice is empty (which means that there is no data left,
    /// since EOF is reached).
    ///
    /// # Errors
    ///
    /// This function will return an I/O error if a `Read` method was called, but returned an error.
    ///
    /// Examples
    ///
    /// ```
    /// #![feature(buf_read_has_data_left)]
    /// use std::io;
    /// use std::io::prelude::*;
    ///
    /// let stdin = io::stdin();
    /// let mut stdin = stdin.lock();
    ///
    /// while stdin.has_data_left()? {
    ///     let mut line = String::new();
    ///     stdin.read_line(&mut line)?;
    ///     // work with line
    ///     println!("{line:?}");
    /// }
    /// # std::io::Result::Ok(())
    /// ```
    #[unstable(feature = "buf_read_has_data_left", issue = "86423")]
    fn has_data_left(&mut self) -> Result<bool> {
        self.fill_buf().map(|b| !b.is_empty())
    }

    /// Reads all bytes into `buf` until the delimiter `byte` or EOF is reached.
    ///
    /// This function will read bytes from the underlying stream until the
    /// delimiter or EOF is found. Once found, all bytes up to, and including,
    /// the delimiter (if found) will be appended to `buf`.
    ///
    /// If successful, this function will return the total number of bytes read.
    ///
    /// This function is blocking and should be used carefully: it is possible for
    /// an attacker to continuously send bytes without ever sending the delimiter
    /// or EOF.
    ///
    /// # Errors
    ///
    /// This function will ignore all instances of [`ErrorKind::Interrupted`] and
    /// will otherwise return any errors returned by [`fill_buf`].
    ///
    /// If an I/O error is encountered then all bytes read so far will be
    /// present in `buf` and its length will have been adjusted appropriately.
    ///
    /// [`fill_buf`]: BufRead::fill_buf
    ///
    /// # Examples
    ///
    /// [`std::io::Cursor`][`Cursor`] is a type that implements `BufRead`. In
    /// this example, we use [`Cursor`] to read all the bytes in a byte slice
    /// in hyphen delimited segments:
    ///
    /// ```
    /// use std::io::{self, BufRead};
    ///
    /// let mut cursor = io::Cursor::new(b"lorem-ipsum");
    /// let mut buf = vec![];
    ///
    /// // cursor is at 'l'
    /// let num_bytes = cursor.read_until(b'-', &mut buf)
    ///     .expect("reading from cursor won't fail");
    /// assert_eq!(num_bytes, 6);
    /// assert_eq!(buf, b"lorem-");
    /// buf.clear();
    ///
    /// // cursor is at 'i'
    /// let num_bytes = cursor.read_until(b'-', &mut buf)
    ///     .expect("reading from cursor won't fail");
    /// assert_eq!(num_bytes, 5);
    /// assert_eq!(buf, b"ipsum");
    /// buf.clear();
    ///
    /// // cursor is at EOF
    /// let num_bytes = cursor.read_until(b'-', &mut buf)
    ///     .expect("reading from cursor won't fail");
    /// assert_eq!(num_bytes, 0);
    /// assert_eq!(buf, b"");
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    fn read_until(&mut self, byte: u8, buf: &mut Vec<u8>) -> Result<usize> {
        read_until(self, byte, buf)
    }

    /// Skips all bytes until the delimiter `byte` or EOF is reached.
    ///
    /// This function will read (and discard) bytes from the underlying stream until the
    /// delimiter or EOF is found.
    ///
    /// If successful, this function will return the total number of bytes read,
    /// including the delimiter byte if found.
    ///
    /// This is useful for efficiently skipping data such as NUL-terminated strings
    /// in binary file formats without buffering.
    ///
    /// This function is blocking and should be used carefully: it is possible for
    /// an attacker to continuously send bytes without ever sending the delimiter
    /// or EOF.
    ///
    /// # Errors
    ///
    /// This function will ignore all instances of [`ErrorKind::Interrupted`] and
    /// will otherwise return any errors returned by [`fill_buf`].
    ///
    /// If an I/O error is encountered then all bytes read so far will be
    /// present in `buf` and its length will have been adjusted appropriately.
    ///
    /// [`fill_buf`]: BufRead::fill_buf
    ///
    /// # Examples
    ///
    /// [`std::io::Cursor`][`Cursor`] is a type that implements `BufRead`. In
    /// this example, we use [`Cursor`] to read some NUL-terminated information
    /// about Ferris from a binary string, skipping the fun fact:
    ///
    /// ```
    /// use std::io::{self, BufRead};
    ///
    /// let mut cursor = io::Cursor::new(b"Ferris\0Likes long walks on the beach\0Crustacean\0!");
    ///
    /// // read name
    /// let mut name = Vec::new();
    /// let num_bytes = cursor.read_until(b'\0', &mut name)
    ///     .expect("reading from cursor won't fail");
    /// assert_eq!(num_bytes, 7);
    /// assert_eq!(name, b"Ferris\0");
    ///
    /// // skip fun fact
    /// let num_bytes = cursor.skip_until(b'\0')
    ///     .expect("reading from cursor won't fail");
    /// assert_eq!(num_bytes, 30);
    ///
    /// // read animal type
    /// let mut animal = Vec::new();
    /// let num_bytes = cursor.read_until(b'\0', &mut animal)
    ///     .expect("reading from cursor won't fail");
    /// assert_eq!(num_bytes, 11);
    /// assert_eq!(animal, b"Crustacean\0");
    ///
    /// // reach EOF
    /// let num_bytes = cursor.skip_until(b'\0')
    ///     .expect("reading from cursor won't fail");
    /// assert_eq!(num_bytes, 1);
    /// ```
    #[stable(feature = "bufread_skip_until", since = "1.83.0")]
    fn skip_until(&mut self, byte: u8) -> Result<usize> {
        skip_until(self, byte)
    }

    /// Reads all bytes until a newline (the `0xA` byte) is reached, and append
    /// them to the provided `String` buffer.
    ///
    /// Previous content of the buffer will be preserved. To avoid appending to
    /// the buffer, you need to [`clear`] it first.
    ///
    /// This function will read bytes from the underlying stream until the
    /// newline delimiter (the `0xA` byte) or EOF is found. Once found, all bytes
    /// up to, and including, the delimiter (if found) will be appended to
    /// `buf`.
    ///
    /// If successful, this function will return the total number of bytes read.
    ///
    /// If this function returns [`Ok(0)`], the stream has reached EOF.
    ///
    /// This function is blocking and should be used carefully: it is possible for
    /// an attacker to continuously send bytes without ever sending a newline
    /// or EOF. You can use [`take`] to limit the maximum number of bytes read.
    ///
    /// [`Ok(0)`]: Ok
    /// [`clear`]: String::clear
    /// [`take`]: crate::io::Read::take
    ///
    /// # Errors
    ///
    /// This function has the same error semantics as [`read_until`] and will
    /// also return an error if the read bytes are not valid UTF-8. If an I/O
    /// error is encountered then `buf` may contain some bytes already read in
    /// the event that all data read so far was valid UTF-8.
    ///
    /// [`read_until`]: BufRead::read_until
    ///
    /// # Examples
    ///
    /// [`std::io::Cursor`][`Cursor`] is a type that implements `BufRead`. In
    /// this example, we use [`Cursor`] to read all the lines in a byte slice:
    ///
    /// ```
    /// use std::io::{self, BufRead};
    ///
    /// let mut cursor = io::Cursor::new(b"foo\nbar");
    /// let mut buf = String::new();
    ///
    /// // cursor is at 'f'
    /// let num_bytes = cursor.read_line(&mut buf)
    ///     .expect("reading from cursor won't fail");
    /// assert_eq!(num_bytes, 4);
    /// assert_eq!(buf, "foo\n");
    /// buf.clear();
    ///
    /// // cursor is at 'b'
    /// let num_bytes = cursor.read_line(&mut buf)
    ///     .expect("reading from cursor won't fail");
    /// assert_eq!(num_bytes, 3);
    /// assert_eq!(buf, "bar");
    /// buf.clear();
    ///
    /// // cursor is at EOF
    /// let num_bytes = cursor.read_line(&mut buf)
    ///     .expect("reading from cursor won't fail");
    /// assert_eq!(num_bytes, 0);
    /// assert_eq!(buf, "");
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    fn read_line(&mut self, buf: &mut String) -> Result<usize> {
        // Note that we are not calling the `.read_until` method here, but
        // rather our hardcoded implementation. For more details as to why, see
        // the comments in `default_read_to_string`.
        unsafe { append_to_string(buf, |b| read_until(self, b'\n', b)) }
    }

    /// Returns an iterator over the contents of this reader split on the byte
    /// `byte`.
    ///
    /// The iterator returned from this function will return instances of
    /// <code>[io::Result]<[Vec]\<u8>></code>. Each vector returned will *not* have
    /// the delimiter byte at the end.
    ///
    /// This function will yield errors whenever [`read_until`] would have
    /// also yielded an error.
    ///
    /// [io::Result]: self::Result "io::Result"
    /// [`read_until`]: BufRead::read_until
    ///
    /// # Examples
    ///
    /// [`std::io::Cursor`][`Cursor`] is a type that implements `BufRead`. In
    /// this example, we use [`Cursor`] to iterate over all hyphen delimited
    /// segments in a byte slice
    ///
    /// ```
    /// use std::io::{self, BufRead};
    ///
    /// let cursor = io::Cursor::new(b"lorem-ipsum-dolor");
    ///
    /// let mut split_iter = cursor.split(b'-').map(|l| l.unwrap());
    /// assert_eq!(split_iter.next(), Some(b"lorem".to_vec()));
    /// assert_eq!(split_iter.next(), Some(b"ipsum".to_vec()));
    /// assert_eq!(split_iter.next(), Some(b"dolor".to_vec()));
    /// assert_eq!(split_iter.next(), None);
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    fn split(self, byte: u8) -> Split<Self>
    where
        Self: Sized,
    {
        Split { buf: self, delim: byte }
    }

    /// Returns an iterator over the lines of this reader.
    ///
    /// The iterator returned from this function will yield instances of
    /// <code>[io::Result]<[String]></code>. Each string returned will *not* have a newline
    /// byte (the `0xA` byte) or `CRLF` (`0xD`, `0xA` bytes) at the end.
    ///
    /// [io::Result]: self::Result "io::Result"
    ///
    /// # Examples
    ///
    /// [`std::io::Cursor`][`Cursor`] is a type that implements `BufRead`. In
    /// this example, we use [`Cursor`] to iterate over all the lines in a byte
    /// slice.
    ///
    /// ```
    /// use std::io::{self, BufRead};
    ///
    /// let cursor = io::Cursor::new(b"lorem\nipsum\r\ndolor");
    ///
    /// let mut lines_iter = cursor.lines().map(|l| l.unwrap());
    /// assert_eq!(lines_iter.next(), Some(String::from("lorem")));
    /// assert_eq!(lines_iter.next(), Some(String::from("ipsum")));
    /// assert_eq!(lines_iter.next(), Some(String::from("dolor")));
    /// assert_eq!(lines_iter.next(), None);
    /// ```
    ///
    /// # Errors
    ///
    /// Each line of the iterator has the same error semantics as [`BufRead::read_line`].
    #[stable(feature = "rust1", since = "1.0.0")]
    fn lines(self) -> Lines<Self>
    where
        Self: Sized,
    {
        Lines { buf: self }
    }
}

#[stable(feature = "chain_bufread", since = "1.9.0")]
impl<T: BufRead, U: BufRead> BufRead for Chain<T, U> {
    fn fill_buf(&mut self) -> Result<&[u8]> {
        if !self.done_first {
            match self.first.fill_buf()? {
                buf if buf.is_empty() => self.done_first = true,
                buf => return Ok(buf),
            }
        }
        self.second.fill_buf()
    }

    fn consume(&mut self, amt: usize) {
        if !self.done_first { self.first.consume(amt) } else { self.second.consume(amt) }
    }

    fn read_until(&mut self, byte: u8, buf: &mut Vec<u8>) -> Result<usize> {
        let mut read = 0;
        if !self.done_first {
            let n = self.first.read_until(byte, buf)?;
            read += n;

            match buf.last() {
                Some(b) if *b == byte && n != 0 => return Ok(read),
                _ => self.done_first = true,
            }
        }
        read += self.second.read_until(byte, buf)?;
        Ok(read)
    }

    // We don't override `read_line` here because an UTF-8 sequence could be
    // split between the two parts of the chain
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<T: BufRead> BufRead for Take<T> {
    fn fill_buf(&mut self) -> Result<&[u8]> {
        // Don't call into inner reader at all at EOF because it may still block
        if self.limit == 0 {
            return Ok(&[]);
        }

        let buf = self.inner.fill_buf()?;
        let cap = cmp::min(buf.len() as u64, self.limit) as usize;
        Ok(&buf[..cap])
    }

    fn consume(&mut self, amt: usize) {
        // Don't let callers reset the limit by passing an overlarge value
        let amt = cmp::min(amt as u64, self.limit) as usize;
        self.limit -= amt as u64;
        self.inner.consume(amt);
    }
}

/// An iterator over the contents of an instance of `BufRead` split on a
/// particular byte.
///
/// This struct is generally created by calling [`split`] on a `BufRead`.
/// Please see the documentation of [`split`] for more details.
///
/// [`split`]: BufRead::split
#[stable(feature = "rust1", since = "1.0.0")]
#[derive(Debug)]
#[cfg_attr(not(test), rustc_diagnostic_item = "IoSplit")]
pub struct Split<B> {
    buf: B,
    delim: u8,
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<B: BufRead> Iterator for Split<B> {
    type Item = Result<Vec<u8>>;

    fn next(&mut self) -> Option<Result<Vec<u8>>> {
        let mut buf = Vec::new();
        match self.buf.read_until(self.delim, &mut buf) {
            Ok(0) => None,
            Ok(_n) => {
                if buf[buf.len() - 1] == self.delim {
                    buf.pop();
                }
                Some(Ok(buf))
            }
            Err(e) => Some(Err(e)),
        }
    }
}

/// An iterator over the lines of an instance of `BufRead`.
///
/// This struct is generally created by calling [`lines`] on a `BufRead`.
/// Please see the documentation of [`lines`] for more details.
///
/// [`lines`]: BufRead::lines
#[stable(feature = "rust1", since = "1.0.0")]
#[derive(Debug)]
#[cfg_attr(not(test), rustc_diagnostic_item = "IoLines")]
pub struct Lines<B> {
    buf: B,
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<B: BufRead> Iterator for Lines<B> {
    type Item = Result<String>;

    fn next(&mut self) -> Option<Result<String>> {
        let mut buf = String::new();
        match self.buf.read_line(&mut buf) {
            Ok(0) => None,
            Ok(_n) => {
                if buf.ends_with('\n') {
                    buf.pop();
                    if buf.ends_with('\r') {
                        buf.pop();
                    }
                }
                Some(Ok(buf))
            }
            Err(e) => Some(Err(e)),
        }
    }
}
