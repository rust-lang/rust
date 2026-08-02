//! Traits, helpers, and type definitions for core I/O functionality.
//!
//! The `io` module contains a number of common things you'll need
//! when doing input and output. The most core part of this module is
//! the [`Read`] and [`Write`] traits, which provide the
//! most general interface for reading and writing input and output.
//!
//! ## Read and Write
//!
//! Because they are traits, [`Read`] and [`Write`] are implemented by a number
//! of other types, and you can implement them for your types too. As such,
//! you'll see a few different types of I/O throughout the documentation in
//! this module. For example, [`Read`] adds a [`read`][`Read::read`] method
//!  which we can use on byte slices:
//!
//! ```no_run
//! use alloc::io;
//! use alloc::io::prelude::*;
//! use alloc::vec::Vec;
//!
//! fn main() -> io::Result<()> {
//!     let data = (0..).into_iter().take(32).collect::<Vec<u8>>();
//!     let mut buffer = [0; 10];
//!
//!     // read up to 10 bytes
//!     let n = data.as_slice().read(&mut buffer)?;
//!
//!     assert_eq!(&[0, 1, 2, 3, 4, 5, 6, 7, 8, 9][..n], &buffer[..n]);
//!
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
//! use alloc::io;
//! use alloc::io::prelude::*;
//! use alloc::io::SeekFrom;
//! use alloc::vec::Vec;
//!
//! # #[allow(dead_code)]
//! fn read_from_end<T: Read + Seek>(reader: &mut T) -> io::Result<Vec<u8>> {
//!     let mut buffer = Vec::new();
//!
//!     // skip to the last 10 bytes of the file
//!     reader.seek(SeekFrom::End(-10))?;
//!
//!     // read up to 10 bytes
//!     let _n = reader.read(&mut buffer)?;
//!
//!     Ok(buffer)
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
//! `alloc::io` comes with two structs, [`BufReader`] and [`BufWriter`], which wrap
//! readers and writers. The wrapper uses a buffer, reducing the number of
//! calls and providing nicer methods for accessing exactly what you want.
//!
//! For example, [`BufReader`] works with the [`BufRead`] trait to add extra
//! methods to any reader:
//!
//! ```no_run
//! use alloc::io;
//! use alloc::io::BufReader;
//! use alloc::io::prelude::*;
//! use alloc::string::String;
//!
//! # #[allow(dead_code)]
//! fn read_one_line<T: Read>(reader: &mut T) -> io::Result<String> {
//!     // reader now implements BufRead
//!     let mut reader = BufReader::new(reader);
//!
//!     let mut buffer = String::new();
//!
//!     // read a line into buffer
//!     reader.read_line(&mut buffer)?;
//!
//!     Ok(buffer)
//! }
//! ```
//!
//! [`BufWriter`] doesn't add any new ways of writing; it just buffers every call
//! to [`write`][`Write::write`]:
//!
//! ```no_run
//! use alloc::io;
//! use alloc::io::BufWriter;
//! use alloc::io::prelude::*;
//!
//! # #[allow(dead_code)]
//! fn write_the_answer<T: Write>(writer: &mut T) -> io::Result<()> {
//!     {
//!         let mut writer = BufWriter::new(writer);
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
//! ## Iterator types
//!
//! A large number of the structures provided by `alloc::io` are for various
//! ways of iterating over I/O. For example, [`Lines`] is used to split over
//! lines:
//!
//! ```no_run
//! use alloc::io;
//! use alloc::io::BufReader;
//! use alloc::io::prelude::*;
//!
//! # #[allow(dead_code)]
//! fn read_one_line<T: Read>(reader: &mut T) -> io::Result<()> {
//!     let mut reader = BufReader::new(reader);
//!
//!     for line in reader.lines() {
//!         assert!(!line?.ends_with('\n'));
//!     }
//!
//!     Ok(())
//! }
//! ```
//!
//! ## io::Result
//!
//! Last, but certainly not least, is [`io::Result`]. This type is used
//! as the return type of many `std::io` functions that can cause an error, and
//! can be returned from your own functions as well. Many of the examples in this
//! module use the [`?` operator]:
//!
//! ```no_run
//! use alloc::io;
//! use alloc::io::prelude::*;
//!
//! # #[allow(dead_code)]
//! fn read_one_line<T: BufRead>(reader: &mut T) -> io::Result<()> {
//!     for line in reader.lines() {
//!         // Reading a line could fail! We use ? to propagate the error
//!         let line = line?;
//!
//!         assert!(!line.ends_with('\n'));
//!     }
//!
//!     Ok(())
//! }
//! ```
//!
//! The return type of `read_one_line()`, [`io::Result<()>`][`io::Result`], is a very
//! common type for functions which don't have a 'real' return value, but do want to
//! return errors if they happen. In this case, the only purpose of this function is
//! to read the lines, so we use `()`.
//!
//! [`Vec<T>`]: crate::vec::Vec
//! [`io::Result`]: self::Result
//! [`?` operator]: ../../book/appendix-02-operators.html

mod buf_read;
mod buffered;
mod copy;
mod cursor;
mod error;
mod impls;
#[unstable(feature = "alloc_io", issue = "154046")]
pub mod prelude;
mod read;
mod util;

#[unstable(feature = "raw_os_error_ty", issue = "107792")]
pub use core::io::RawOsError;
#[unstable(feature = "io_const_error_internals", issue = "none")]
pub use core::io::SimpleMessage;
#[unstable(feature = "io_const_error", issue = "133448")]
pub use core::io::const_error;
#[unstable(feature = "core_io_borrowed_buf", issue = "117693")]
pub use core::io::{BorrowedBuf, BorrowedCursor};
#[unstable(feature = "alloc_io", issue = "154046")]
pub use core::io::{
    Chain, Cursor, Empty, Error, ErrorKind, IoSlice, IoSliceMut, Repeat, Result, Seek, SeekFrom,
    Sink, Take, Write, empty, repeat, sink,
};
#[doc(hidden)]
#[unstable(feature = "core_io_internals", reason = "exposed only for libstd", issue = "none")]
pub use core::io::{IoHandle, OsFunctions, default_write_vectored, stream_len_default};
use core::io::{
    SizeHint, WriteThroughCursor, chain, slice_write, slice_write_all, slice_write_all_vectored,
    slice_write_vectored, take,
};

use self::read::{append_to_string, default_read_buf_exact, default_read_exact};
use self::util::{bytes, lines, split, uninlined_slow_read_byte};
#[unstable(feature = "alloc_io", issue = "154046")]
pub use self::{
    buf_read::BufRead,
    buffered::{BufReader, BufWriter, IntoInnerError, LineWriter, WriterPanicked},
    copy::copy,
    read::{Read, read_to_string},
    util::{Bytes, Lines, Split},
};
#[doc(hidden)]
#[unstable(feature = "core_io_internals", reason = "exposed only for libstd", issue = "none")]
pub use self::{
    copy::{CopyState, SpecCopy},
    read::{
        DEFAULT_BUF_SIZE, default_read_buf, default_read_to_end, default_read_to_string,
        default_read_vectored,
    },
    util::SpecReadByte,
};
