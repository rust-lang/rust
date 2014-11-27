#![experimental]
#![deny(unused_must_use)]

pub use self::SeekStyle::*;
pub use self::FileMode::*;
pub use self::FileAccess::*;
pub use self::FileType::*;
pub use self::IoErrorKind::*;

use char::Char;
use clone::Clone;
use default::Default;
use error::{FromError, Error};
use fmt;
use int;
use iter::Iterator;
use mem::transmute;
use ops::{BitOr, BitXor, BitAnd, Sub, Not};
use option::{Option, Some, None};
use os;
use boxed::Box;
use result::{Ok, Err, Result};
use sys;
use slice::{AsSlice, SlicePrelude};
use str::{Str, StrPrelude};
use str;
use string::String;
use uint;
use unicode::char::UnicodeChar;
use vec::Vec;

// Reexports
pub use self::stdio::stdin;
pub use self::stdio::stdout;
pub use self::stdio::stderr;
pub use self::stdio::print;
pub use self::stdio::println;

pub use self::fs::File;
pub use self::timer::Timer;
pub use self::net::ip::IpAddr;
pub use self::net::tcp::TcpListener;
pub use self::net::tcp::TcpStream;
pub use self::net::udp::UdpStream;
pub use self::pipe::PipeStream;
pub use self::process::{Process, Command};
pub use self::tempfile::TempDir;

pub use self::mem::{MemReader, BufReader, MemWriter, BufWriter};
pub use self::buffered::{BufferedReader, BufferedWriter, BufferedStream,
                         LineBufferedWriter};
pub use self::comm_adapters::{ChanReader, ChanWriter};

mod buffered;
mod comm_adapters;
mod mem;
mod result;
mod tempfile;
pub mod extensions;
pub mod fs;
pub mod net;
pub mod pipe;
pub mod process;
pub mod stdio;
pub mod test;
pub mod timer;
pub mod util;

/// The default buffer size for various I/O operations
// libuv recommends 64k buffers to maximize throughput
// https://groups.google.com/forum/#!topic/libuv/oQO1HJAIDdA
const DEFAULT_BUF_SIZE: uint = 1024 * 64;

/// A convenient typedef of the return value of any I/O action.
pub type IoResult<T> = Result<T, IoError>;

/// The type passed to I/O condition handlers to indicate error
///
/// # FIXME
///
/// Is something like this sufficient? It's kind of archaic
#[deriving(PartialEq, Eq, Clone)]
pub struct IoError {
    /// An enumeration which can be matched against for determining the flavor
    /// of error.
    pub kind: IoErrorKind,
    /// A human-readable description about the error
    pub desc: &'static str,
    /// Detailed information about this error, not always available
    pub detail: Option<String>
}

impl IoError {
    /// Convert an `errno` value into an `IoError`.
    ///
    /// If `detail` is `true`, the `detail` field of the `IoError`
    /// struct is filled with an allocated string describing the error
    /// in more detail, retrieved from the operating system.
    pub fn from_errno(errno: uint, detail: bool) -> IoError { unimplemented!() }

    /// Retrieve the last error to occur as a (detailed) IoError.
    ///
    /// This uses the OS `errno`, and so there should not be any task
    /// descheduling or migration (other than that performed by the
    /// operating system) between the call(s) for which errors are
    /// being checked and the call of this function.
    pub fn last_error() -> IoError { unimplemented!() }
}

impl fmt::Show for IoError {
    fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result { unimplemented!() }
}

impl Error for IoError {
    fn description(&self) -> &str { unimplemented!() }

    fn detail(&self) -> Option<String> { unimplemented!() }
}

impl FromError<IoError> for Box<Error> {
    fn from_error(err: IoError) -> Box<Error> { unimplemented!() }
}

/// A list specifying general categories of I/O error.
#[deriving(PartialEq, Eq, Clone, Show)]
pub enum IoErrorKind {
    /// Any I/O error not part of this list.
    OtherIoError,
    /// The operation could not complete because end of file was reached.
    EndOfFile,
    /// The file was not found.
    FileNotFound,
    /// The file permissions disallowed access to this file.
    PermissionDenied,
    /// A network connection failed for some reason not specified in this list.
    ConnectionFailed,
    /// The network operation failed because the network connection was closed.
    Closed,
    /// The connection was refused by the remote server.
    ConnectionRefused,
    /// The connection was reset by the remote server.
    ConnectionReset,
    /// The connection was aborted (terminated) by the remote server.
    ConnectionAborted,
    /// The network operation failed because it was not connected yet.
    NotConnected,
    /// The operation failed because a pipe was closed.
    BrokenPipe,
    /// A file already existed with that name.
    PathAlreadyExists,
    /// No file exists at that location.
    PathDoesntExist,
    /// The path did not specify the type of file that this operation required. For example,
    /// attempting to copy a directory with the `fs::copy()` operation will fail with this error.
    MismatchedFileTypeForOperation,
    /// The operation temporarily failed (for example, because a signal was received), and retrying
    /// may succeed.
    ResourceUnavailable,
    /// No I/O functionality is available for this task.
    IoUnavailable,
    /// A parameter was incorrect in a way that caused an I/O error not part of this list.
    InvalidInput,
    /// The I/O operation's timeout expired, causing it to be canceled.
    TimedOut,
    /// This write operation failed to write all of its data.
    ///
    /// Normally the write() method on a Writer guarantees that all of its data
    /// has been written, but some operations may be terminated after only
    /// partially writing some data. An example of this is a timed out write
    /// which successfully wrote a known number of bytes, but bailed out after
    /// doing so.
    ///
    /// The payload contained as part of this variant is the number of bytes
    /// which are known to have been successfully written.
    ShortWrite(uint),
    /// The Reader returned 0 bytes from `read()` too many times.
    NoProgress,
}

/// A trait that lets you add a `detail` to an IoError easily
trait UpdateIoError<T> {
    /// Returns an IoError with updated description and detail
    fn update_err(self, desc: &'static str, detail: |&IoError| -> String) -> Self;

    /// Returns an IoError with updated detail
    fn update_detail(self, detail: |&IoError| -> String) -> Self;

    /// Returns an IoError with update description
    fn update_desc(self, desc: &'static str) -> Self;
}

impl<T> UpdateIoError<T> for IoResult<T> {
    fn update_err(self, desc: &'static str, detail: |&IoError| -> String) -> IoResult<T> { unimplemented!() }

    fn update_detail(self, detail: |&IoError| -> String) -> IoResult<T> { unimplemented!() }

    fn update_desc(self, desc: &'static str) -> IoResult<T> { unimplemented!() }
}

static NO_PROGRESS_LIMIT: uint = 1000;

/// A trait for objects which are byte-oriented streams. Readers are defined by
/// one method, `read`. This function will block until data is available,
/// filling in the provided buffer with any data read.
///
/// Readers are intended to be composable with one another. Many objects
/// throughout the I/O and related libraries take and provide types which
/// implement the `Reader` trait.
pub trait Reader {

    // Only method which need to get implemented for this trait

    /// Read bytes, up to the length of `buf` and place them in `buf`.
    /// Returns the number of bytes read. The number of bytes read may
    /// be less than the number requested, even 0. Returns `Err` on EOF.
    ///
    /// # Error
    ///
    /// If an error occurs during this I/O operation, then it is returned as
    /// `Err(IoError)`. Note that end-of-file is considered an error, and can be
    /// inspected for in the error's `kind` field. Also note that reading 0
    /// bytes is not considered an error in all circumstances
    ///
    /// # Implementation Note
    ///
    /// When implementing this method on a new Reader, you are strongly encouraged
    /// not to return 0 if you can avoid it.
    fn read(&mut self, buf: &mut [u8]) -> IoResult<uint>;

    // Convenient helper methods based on the above methods

    /// Reads at least `min` bytes and places them in `buf`.
    /// Returns the number of bytes read.
    ///
    /// This will continue to call `read` until at least `min` bytes have been
    /// read. If `read` returns 0 too many times, `NoProgress` will be
    /// returned.
    ///
    /// # Error
    ///
    /// If an error occurs at any point, that error is returned, and no further
    /// bytes are read.
    fn read_at_least(&mut self, min: uint, buf: &mut [u8]) -> IoResult<uint> { unimplemented!() }

    /// Reads a single byte. Returns `Err` on EOF.
    fn read_byte(&mut self) -> IoResult<u8> { unimplemented!() }

    /// Reads up to `len` bytes and appends them to a vector.
    /// Returns the number of bytes read. The number of bytes read may be
    /// less than the number requested, even 0. Returns Err on EOF.
    ///
    /// # Error
    ///
    /// If an error occurs during this I/O operation, then it is returned
    /// as `Err(IoError)`. See `read()` for more details.
    fn push(&mut self, len: uint, buf: &mut Vec<u8>) -> IoResult<uint> { unimplemented!() }

    /// Reads at least `min` bytes, but no more than `len`, and appends them to
    /// a vector.
    /// Returns the number of bytes read.
    ///
    /// This will continue to call `read` until at least `min` bytes have been
    /// read. If `read` returns 0 too many times, `NoProgress` will be
    /// returned.
    ///
    /// # Error
    ///
    /// If an error occurs at any point, that error is returned, and no further
    /// bytes are read.
    fn push_at_least(&mut self, min: uint, len: uint, buf: &mut Vec<u8>) -> IoResult<uint> { unimplemented!() }

    /// Reads exactly `len` bytes and gives you back a new vector of length
    /// `len`
    ///
    /// # Error
    ///
    /// Fails with the same conditions as `read`. Additionally returns error
    /// on EOF. Note that if an error is returned, then some number of bytes may
    /// have already been consumed from the underlying reader, and they are lost
    /// (not returned as part of the error). If this is unacceptable, then it is
    /// recommended to use the `push_at_least` or `read` methods.
    fn read_exact(&mut self, len: uint) -> IoResult<Vec<u8>> { unimplemented!() }

    /// Reads all remaining bytes from the stream.
    ///
    /// # Error
    ///
    /// Returns any non-EOF error immediately. Previously read bytes are
    /// discarded when an error is returned.
    ///
    /// When EOF is encountered, all bytes read up to that point are returned.
    fn read_to_end(&mut self) -> IoResult<Vec<u8>> { unimplemented!() }

    /// Reads all of the remaining bytes of this stream, interpreting them as a
    /// UTF-8 encoded stream. The corresponding string is returned.
    ///
    /// # Error
    ///
    /// This function returns all of the same errors as `read_to_end` with an
    /// additional error if the reader's contents are not a valid sequence of
    /// UTF-8 bytes.
    fn read_to_string(&mut self) -> IoResult<String> { unimplemented!() }

    // Byte conversion helpers

    /// Reads `n` little-endian unsigned integer bytes.
    ///
    /// `n` must be between 1 and 8, inclusive.
    fn read_le_uint_n(&mut self, nbytes: uint) -> IoResult<u64> { unimplemented!() }

    /// Reads `n` little-endian signed integer bytes.
    ///
    /// `n` must be between 1 and 8, inclusive.
    fn read_le_int_n(&mut self, nbytes: uint) -> IoResult<i64> { unimplemented!() }

    /// Reads `n` big-endian unsigned integer bytes.
    ///
    /// `n` must be between 1 and 8, inclusive.
    fn read_be_uint_n(&mut self, nbytes: uint) -> IoResult<u64> { unimplemented!() }

    /// Reads `n` big-endian signed integer bytes.
    ///
    /// `n` must be between 1 and 8, inclusive.
    fn read_be_int_n(&mut self, nbytes: uint) -> IoResult<i64> { unimplemented!() }

    /// Reads a little-endian unsigned integer.
    ///
    /// The number of bytes returned is system-dependent.
    fn read_le_uint(&mut self) -> IoResult<uint> { unimplemented!() }

    /// Reads a little-endian integer.
    ///
    /// The number of bytes returned is system-dependent.
    fn read_le_int(&mut self) -> IoResult<int> { unimplemented!() }

    /// Reads a big-endian unsigned integer.
    ///
    /// The number of bytes returned is system-dependent.
    fn read_be_uint(&mut self) -> IoResult<uint> { unimplemented!() }

    /// Reads a big-endian integer.
    ///
    /// The number of bytes returned is system-dependent.
    fn read_be_int(&mut self) -> IoResult<int> { unimplemented!() }

    /// Reads a big-endian `u64`.
    ///
    /// `u64`s are 8 bytes long.
    fn read_be_u64(&mut self) -> IoResult<u64> { unimplemented!() }

    /// Reads a big-endian `u32`.
    ///
    /// `u32`s are 4 bytes long.
    fn read_be_u32(&mut self) -> IoResult<u32> { unimplemented!() }

    /// Reads a big-endian `u16`.
    ///
    /// `u16`s are 2 bytes long.
    fn read_be_u16(&mut self) -> IoResult<u16> { unimplemented!() }

    /// Reads a big-endian `i64`.
    ///
    /// `i64`s are 8 bytes long.
    fn read_be_i64(&mut self) -> IoResult<i64> { unimplemented!() }

    /// Reads a big-endian `i32`.
    ///
    /// `i32`s are 4 bytes long.
    fn read_be_i32(&mut self) -> IoResult<i32> { unimplemented!() }

    /// Reads a big-endian `i16`.
    ///
    /// `i16`s are 2 bytes long.
    fn read_be_i16(&mut self) -> IoResult<i16> { unimplemented!() }

    /// Reads a big-endian `f64`.
    ///
    /// `f64`s are 8 byte, IEEE754 double-precision floating point numbers.
    fn read_be_f64(&mut self) -> IoResult<f64> { unimplemented!() }

    /// Reads a big-endian `f32`.
    ///
    /// `f32`s are 4 byte, IEEE754 single-precision floating point numbers.
    fn read_be_f32(&mut self) -> IoResult<f32> { unimplemented!() }

    /// Reads a little-endian `u64`.
    ///
    /// `u64`s are 8 bytes long.
    fn read_le_u64(&mut self) -> IoResult<u64> { unimplemented!() }

    /// Reads a little-endian `u32`.
    ///
    /// `u32`s are 4 bytes long.
    fn read_le_u32(&mut self) -> IoResult<u32> { unimplemented!() }

    /// Reads a little-endian `u16`.
    ///
    /// `u16`s are 2 bytes long.
    fn read_le_u16(&mut self) -> IoResult<u16> { unimplemented!() }

    /// Reads a little-endian `i64`.
    ///
    /// `i64`s are 8 bytes long.
    fn read_le_i64(&mut self) -> IoResult<i64> { unimplemented!() }

    /// Reads a little-endian `i32`.
    ///
    /// `i32`s are 4 bytes long.
    fn read_le_i32(&mut self) -> IoResult<i32> { unimplemented!() }

    /// Reads a little-endian `i16`.
    ///
    /// `i16`s are 2 bytes long.
    fn read_le_i16(&mut self) -> IoResult<i16> { unimplemented!() }

    /// Reads a little-endian `f64`.
    ///
    /// `f64`s are 8 byte, IEEE754 double-precision floating point numbers.
    fn read_le_f64(&mut self) -> IoResult<f64> { unimplemented!() }

    /// Reads a little-endian `f32`.
    ///
    /// `f32`s are 4 byte, IEEE754 single-precision floating point numbers.
    fn read_le_f32(&mut self) -> IoResult<f32> { unimplemented!() }

    /// Read a u8.
    ///
    /// `u8`s are 1 byte.
    fn read_u8(&mut self) -> IoResult<u8> { unimplemented!() }

    /// Read an i8.
    ///
    /// `i8`s are 1 byte.
    fn read_i8(&mut self) -> IoResult<i8> { unimplemented!() }
}

/// A reader which can be converted to a RefReader.
#[deprecated = "use ByRefReader instead"]
pub trait AsRefReader {
    /// Creates a wrapper around a mutable reference to the reader.
    ///
    /// This is useful to allow applying adaptors while still
    /// retaining ownership of the original value.
    fn by_ref<'a>(&'a mut self) -> RefReader<'a, Self>;
}

#[allow(deprecated)]
impl<T: Reader> AsRefReader for T {
    fn by_ref<'a>(&'a mut self) -> RefReader<'a, T> { unimplemented!() }
}

/// A reader which can be converted to a RefReader.
pub trait ByRefReader {
    /// Creates a wrapper around a mutable reference to the reader.
    ///
    /// This is useful to allow applying adaptors while still
    /// retaining ownership of the original value.
    fn by_ref<'a>(&'a mut self) -> RefReader<'a, Self>;
}

impl<T: Reader> ByRefReader for T {
    fn by_ref<'a>(&'a mut self) -> RefReader<'a, T> { unimplemented!() }
}

/// A reader which can be converted to bytes.
pub trait BytesReader {
    /// Create an iterator that reads a single byte on
    /// each iteration, until EOF.
    ///
    /// # Error
    ///
    /// Any error other than `EndOfFile` that is produced by the underlying Reader
    /// is returned by the iterator and should be handled by the caller.
    fn bytes<'r>(&'r mut self) -> extensions::Bytes<'r, Self>;
}

impl<T: Reader> BytesReader for T {
    fn bytes<'r>(&'r mut self) -> extensions::Bytes<'r, T> { unimplemented!() }
}

impl<'a> Reader for Box<Reader+'a> {
    fn read(&mut self, buf: &mut [u8]) -> IoResult<uint> { unimplemented!() }
}

impl<'a> Reader for &'a mut Reader+'a {
    fn read(&mut self, buf: &mut [u8]) -> IoResult<uint> { unimplemented!() }
}

/// Returns a slice of `v` between `start` and `end`.
///
/// Similar to `slice()` except this function only bounds the slice on the
/// capacity of `v`, not the length.
///
/// # Panics
///
/// Panics when `start` or `end` point outside the capacity of `v`, or when
/// `start` > `end`.
// Private function here because we aren't sure if we want to expose this as
// API yet. If so, it should be a method on Vec.
unsafe fn slice_vec_capacity<'a, T>(v: &'a mut Vec<T>, start: uint, end: uint) -> &'a mut [T] { unimplemented!() }

/// A `RefReader` is a struct implementing `Reader` which contains a reference
/// to another reader. This is often useful when composing streams.
///
/// # Example
///
/// ```
/// # fn main() {}
/// # fn process_input<R: Reader>(r: R) {}
/// # fn foo() {
/// use std::io;
/// use std::io::ByRefReader;
/// use std::io::util::LimitReader;
///
/// let mut stream = io::stdin();
///
/// // Only allow the function to process at most one kilobyte of input
/// {
///     let stream = LimitReader::new(stream.by_ref(), 1024);
///     process_input(stream);
/// }
///
/// // 'stream' is still available for use here
///
/// # }
/// ```
pub struct RefReader<'a, R:'a> {
    /// The underlying reader which this is referencing
    inner: &'a mut R
}

impl<'a, R: Reader> Reader for RefReader<'a, R> {
    fn read(&mut self, buf: &mut [u8]) -> IoResult<uint> { unimplemented!() }
}

impl<'a, R: Buffer> Buffer for RefReader<'a, R> {
    fn fill_buf<'a>(&'a mut self) -> IoResult<&'a [u8]> { unimplemented!() }
    fn consume(&mut self, amt: uint) { unimplemented!() }
}

fn extend_sign(val: u64, nbytes: uint) -> i64 { unimplemented!() }

/// A trait for objects which are byte-oriented streams. Writers are defined by
/// one method, `write`. This function will block until the provided buffer of
/// bytes has been entirely written, and it will return any failures which occur.
///
/// Another commonly overridden method is the `flush` method for writers such as
/// buffered writers.
///
/// Writers are intended to be composable with one another. Many objects
/// throughout the I/O and related libraries take and provide types which
/// implement the `Writer` trait.
pub trait Writer {
    /// Write the entirety of a given buffer
    ///
    /// # Errors
    ///
    /// If an error happens during the I/O operation, the error is returned as
    /// `Err`. Note that it is considered an error if the entire buffer could
    /// not be written, and if an error is returned then it is unknown how much
    /// data (if any) was actually written.
    fn write(&mut self, buf: &[u8]) -> IoResult<()>;

    /// Flush this output stream, ensuring that all intermediately buffered
    /// contents reach their destination.
    ///
    /// This is by default a no-op and implementers of the `Writer` trait should
    /// decide whether their stream needs to be buffered or not.
    fn flush(&mut self) -> IoResult<()> { unimplemented!() }

    /// Writes a formatted string into this writer, returning any error
    /// encountered.
    ///
    /// This method is primarily used to interface with the `format_args!`
    /// macro, but it is rare that this should explicitly be called. The
    /// `write!` macro should be favored to invoke this method instead.
    ///
    /// # Errors
    ///
    /// This function will return any I/O error reported while formatting.
    fn write_fmt(&mut self, fmt: &fmt::Arguments) -> IoResult<()> { unimplemented!() }

    /// Write a rust string into this sink.
    ///
    /// The bytes written will be the UTF-8 encoded version of the input string.
    /// If other encodings are desired, it is recommended to compose this stream
    /// with another performing the conversion, or to use `write` with a
    /// converted byte-array instead.
    #[inline]
    fn write_str(&mut self, s: &str) -> IoResult<()> { unimplemented!() }

    /// Writes a string into this sink, and then writes a literal newline (`\n`)
    /// byte afterwards. Note that the writing of the newline is *not* atomic in
    /// the sense that the call to `write` is invoked twice (once with the
    /// string and once with a newline character).
    ///
    /// If other encodings or line ending flavors are desired, it is recommended
    /// that the `write` method is used specifically instead.
    #[inline]
    fn write_line(&mut self, s: &str) -> IoResult<()> { unimplemented!() }

    /// Write a single char, encoded as UTF-8.
    #[inline]
    fn write_char(&mut self, c: char) -> IoResult<()> { unimplemented!() }

    /// Write the result of passing n through `int::to_str_bytes`.
    #[inline]
    fn write_int(&mut self, n: int) -> IoResult<()> { unimplemented!() }

    /// Write the result of passing n through `uint::to_str_bytes`.
    #[inline]
    fn write_uint(&mut self, n: uint) -> IoResult<()> { unimplemented!() }

    /// Write a little-endian uint (number of bytes depends on system).
    #[inline]
    fn write_le_uint(&mut self, n: uint) -> IoResult<()> { unimplemented!() }

    /// Write a little-endian int (number of bytes depends on system).
    #[inline]
    fn write_le_int(&mut self, n: int) -> IoResult<()> { unimplemented!() }

    /// Write a big-endian uint (number of bytes depends on system).
    #[inline]
    fn write_be_uint(&mut self, n: uint) -> IoResult<()> { unimplemented!() }

    /// Write a big-endian int (number of bytes depends on system).
    #[inline]
    fn write_be_int(&mut self, n: int) -> IoResult<()> { unimplemented!() }

    /// Write a big-endian u64 (8 bytes).
    #[inline]
    fn write_be_u64(&mut self, n: u64) -> IoResult<()> { unimplemented!() }

    /// Write a big-endian u32 (4 bytes).
    #[inline]
    fn write_be_u32(&mut self, n: u32) -> IoResult<()> { unimplemented!() }

    /// Write a big-endian u16 (2 bytes).
    #[inline]
    fn write_be_u16(&mut self, n: u16) -> IoResult<()> { unimplemented!() }

    /// Write a big-endian i64 (8 bytes).
    #[inline]
    fn write_be_i64(&mut self, n: i64) -> IoResult<()> { unimplemented!() }

    /// Write a big-endian i32 (4 bytes).
    #[inline]
    fn write_be_i32(&mut self, n: i32) -> IoResult<()> { unimplemented!() }

    /// Write a big-endian i16 (2 bytes).
    #[inline]
    fn write_be_i16(&mut self, n: i16) -> IoResult<()> { unimplemented!() }

    /// Write a big-endian IEEE754 double-precision floating-point (8 bytes).
    #[inline]
    fn write_be_f64(&mut self, f: f64) -> IoResult<()> { unimplemented!() }

    /// Write a big-endian IEEE754 single-precision floating-point (4 bytes).
    #[inline]
    fn write_be_f32(&mut self, f: f32) -> IoResult<()> { unimplemented!() }

    /// Write a little-endian u64 (8 bytes).
    #[inline]
    fn write_le_u64(&mut self, n: u64) -> IoResult<()> { unimplemented!() }

    /// Write a little-endian u32 (4 bytes).
    #[inline]
    fn write_le_u32(&mut self, n: u32) -> IoResult<()> { unimplemented!() }

    /// Write a little-endian u16 (2 bytes).
    #[inline]
    fn write_le_u16(&mut self, n: u16) -> IoResult<()> { unimplemented!() }

    /// Write a little-endian i64 (8 bytes).
    #[inline]
    fn write_le_i64(&mut self, n: i64) -> IoResult<()> { unimplemented!() }

    /// Write a little-endian i32 (4 bytes).
    #[inline]
    fn write_le_i32(&mut self, n: i32) -> IoResult<()> { unimplemented!() }

    /// Write a little-endian i16 (2 bytes).
    #[inline]
    fn write_le_i16(&mut self, n: i16) -> IoResult<()> { unimplemented!() }

    /// Write a little-endian IEEE754 double-precision floating-point
    /// (8 bytes).
    #[inline]
    fn write_le_f64(&mut self, f: f64) -> IoResult<()> { unimplemented!() }

    /// Write a little-endian IEEE754 single-precision floating-point
    /// (4 bytes).
    #[inline]
    fn write_le_f32(&mut self, f: f32) -> IoResult<()> { unimplemented!() }

    /// Write a u8 (1 byte).
    #[inline]
    fn write_u8(&mut self, n: u8) -> IoResult<()> { unimplemented!() }

    /// Write an i8 (1 byte).
    #[inline]
    fn write_i8(&mut self, n: i8) -> IoResult<()> { unimplemented!() }
}

/// A writer which can be converted to a RefWriter.
#[deprecated = "use ByRefWriter instead"]
pub trait AsRefWriter {
    /// Creates a wrapper around a mutable reference to the writer.
    ///
    /// This is useful to allow applying wrappers while still
    /// retaining ownership of the original value.
    #[inline]
    fn by_ref<'a>(&'a mut self) -> RefWriter<'a, Self>;
}

#[allow(deprecated)]
impl<T: Writer> AsRefWriter for T {
    fn by_ref<'a>(&'a mut self) -> RefWriter<'a, T> { unimplemented!() }
}

/// A writer which can be converted to a RefWriter.
pub trait ByRefWriter {
    /// Creates a wrapper around a mutable reference to the writer.
    ///
    /// This is useful to allow applying wrappers while still
    /// retaining ownership of the original value.
    #[inline]
    fn by_ref<'a>(&'a mut self) -> RefWriter<'a, Self>;
}

impl<T: Writer> ByRefWriter for T {
    fn by_ref<'a>(&'a mut self) -> RefWriter<'a, T> {
        RefWriter { inner: self }
    }
}

impl<'a> Writer for Box<Writer+'a> {
    #[inline]
    fn write(&mut self, buf: &[u8]) -> IoResult<()> { unimplemented!() }

    #[inline]
    fn flush(&mut self) -> IoResult<()> { unimplemented!() }
}

impl<'a> Writer for &'a mut Writer+'a {
    #[inline]
    fn write(&mut self, buf: &[u8]) -> IoResult<()> { unimplemented!() }

    #[inline]
    fn flush(&mut self) -> IoResult<()> { unimplemented!() }
}

/// A `RefWriter` is a struct implementing `Writer` which contains a reference
/// to another writer. This is often useful when composing streams.
///
/// # Example
///
/// ```
/// # fn main() {}
/// # fn process_input<R: Reader>(r: R) {}
/// # fn foo () {
/// use std::io::util::TeeReader;
/// use std::io::{stdin, ByRefWriter};
///
/// let mut output = Vec::new();
///
/// {
///     // Don't give ownership of 'output' to the 'tee'. Instead we keep a
///     // handle to it in the outer scope
///     let mut tee = TeeReader::new(stdin(), output.by_ref());
///     process_input(tee);
/// }
///
/// println!("input processed: {}", output);
/// # }
/// ```
pub struct RefWriter<'a, W:'a> {
    /// The underlying writer which this is referencing
    inner: &'a mut W
}

impl<'a, W: Writer> Writer for RefWriter<'a, W> {
    #[inline]
    fn write(&mut self, buf: &[u8]) -> IoResult<()> { unimplemented!() }

    #[inline]
    fn flush(&mut self) -> IoResult<()> { unimplemented!() }
}


/// A Stream is a readable and a writable object. Data written is typically
/// received by the object which reads receive data from.
pub trait Stream: Reader + Writer { }

impl<T: Reader + Writer> Stream for T {}

/// An iterator that reads a line on each iteration,
/// until `.read_line()` encounters `EndOfFile`.
///
/// # Notes about the Iteration Protocol
///
/// The `Lines` may yield `None` and thus terminate
/// an iteration, but continue to yield elements if iteration
/// is attempted again.
///
/// # Error
///
/// Any error other than `EndOfFile` that is produced by the underlying Reader
/// is returned by the iterator and should be handled by the caller.
pub struct Lines<'r, T:'r> {
    buffer: &'r mut T,
}

impl<'r, T: Buffer> Iterator<IoResult<String>> for Lines<'r, T> {
    fn next(&mut self) -> Option<IoResult<String>> { unimplemented!() }
}

/// An iterator that reads a utf8-encoded character on each iteration,
/// until `.read_char()` encounters `EndOfFile`.
///
/// # Notes about the Iteration Protocol
///
/// The `Chars` may yield `None` and thus terminate
/// an iteration, but continue to yield elements if iteration
/// is attempted again.
///
/// # Error
///
/// Any error other than `EndOfFile` that is produced by the underlying Reader
/// is returned by the iterator and should be handled by the caller.
pub struct Chars<'r, T:'r> {
    buffer: &'r mut T
}

impl<'r, T: Buffer> Iterator<IoResult<char>> for Chars<'r, T> {
    fn next(&mut self) -> Option<IoResult<char>> { unimplemented!() }
}

/// A Buffer is a type of reader which has some form of internal buffering to
/// allow certain kinds of reading operations to be more optimized than others.
/// This type extends the `Reader` trait with a few methods that are not
/// possible to reasonably implement with purely a read interface.
pub trait Buffer: Reader {
    /// Fills the internal buffer of this object, returning the buffer contents.
    /// Note that none of the contents will be "read" in the sense that later
    /// calling `read` may return the same contents.
    ///
    /// The `consume` function must be called with the number of bytes that are
    /// consumed from this buffer returned to ensure that the bytes are never
    /// returned twice.
    ///
    /// # Error
    ///
    /// This function will return an I/O error if the underlying reader was
    /// read, but returned an error. Note that it is not an error to return a
    /// 0-length buffer.
    fn fill_buf<'a>(&'a mut self) -> IoResult<&'a [u8]>;

    /// Tells this buffer that `amt` bytes have been consumed from the buffer,
    /// so they should no longer be returned in calls to `read`.
    fn consume(&mut self, amt: uint);

    /// Reads the next line of input, interpreted as a sequence of UTF-8
    /// encoded Unicode codepoints. If a newline is encountered, then the
    /// newline is contained in the returned string.
    ///
    /// # Example
    ///
    /// ```rust
    /// use std::io;
    ///
    /// let mut reader = io::stdin();
    /// let input = reader.read_line().ok().unwrap_or("nothing".to_string());
    /// ```
    ///
    /// # Error
    ///
    /// This function has the same error semantics as `read_until`:
    ///
    /// * All non-EOF errors will be returned immediately
    /// * If an error is returned previously consumed bytes are lost
    /// * EOF is only returned if no bytes have been read
    /// * Reach EOF may mean that the delimiter is not present in the return
    ///   value
    ///
    /// Additionally, this function can fail if the line of input read is not a
    /// valid UTF-8 sequence of bytes.
    fn read_line(&mut self) -> IoResult<String> { unimplemented!() }

    /// Reads a sequence of bytes leading up to a specified delimiter. Once the
    /// specified byte is encountered, reading ceases and the bytes up to and
    /// including the delimiter are returned.
    ///
    /// # Error
    ///
    /// If any I/O error is encountered other than EOF, the error is immediately
    /// returned. Note that this may discard bytes which have already been read,
    /// and those bytes will *not* be returned. It is recommended to use other
    /// methods if this case is worrying.
    ///
    /// If EOF is encountered, then this function will return EOF if 0 bytes
    /// have been read, otherwise the pending byte buffer is returned. This
    /// is the reason that the byte buffer returned may not always contain the
    /// delimiter.
    fn read_until(&mut self, byte: u8) -> IoResult<Vec<u8>> { unimplemented!() }

    /// Reads the next utf8-encoded character from the underlying stream.
    ///
    /// # Error
    ///
    /// If an I/O error occurs, or EOF, then this function will return `Err`.
    /// This function will also return error if the stream does not contain a
    /// valid utf-8 encoded codepoint as the next few bytes in the stream.
    fn read_char(&mut self) -> IoResult<char> { unimplemented!() }
}

/// Extension methods for the Buffer trait which are included in the prelude.
pub trait BufferPrelude {
    /// Create an iterator that reads a utf8-encoded character on each iteration
    /// until EOF.
    ///
    /// # Error
    ///
    /// Any error other than `EndOfFile` that is produced by the underlying Reader
    /// is returned by the iterator and should be handled by the caller.
    fn chars<'r>(&'r mut self) -> Chars<'r, Self>;

    /// Create an iterator that reads a line on each iteration until EOF.
    ///
    /// # Error
    ///
    /// Any error other than `EndOfFile` that is produced by the underlying Reader
    /// is returned by the iterator and should be handled by the caller.
    fn lines<'r>(&'r mut self) -> Lines<'r, Self>;
}

impl<T: Buffer> BufferPrelude for T {
    fn chars<'r>(&'r mut self) -> Chars<'r, T> { unimplemented!() }

    fn lines<'r>(&'r mut self) -> Lines<'r, T> { unimplemented!() }
}

/// When seeking, the resulting cursor is offset from a base by the offset given
/// to the `seek` function. The base used is specified by this enumeration.
pub enum SeekStyle {
    /// Seek from the beginning of the stream
    SeekSet,
    /// Seek from the end of the stream
    SeekEnd,
    /// Seek from the current position
    SeekCur,
}

/// An object implementing `Seek` internally has some form of cursor which can
/// be moved within a stream of bytes. The stream typically has a fixed size,
/// allowing seeking relative to either end.
pub trait Seek {
    /// Return position of file cursor in the stream
    fn tell(&self) -> IoResult<u64>;

    /// Seek to an offset in a stream
    ///
    /// A successful seek clears the EOF indicator. Seeking beyond EOF is
    /// allowed, but seeking before position 0 is not allowed.
    ///
    /// # Errors
    ///
    /// * Seeking to a negative offset is considered an error
    /// * Seeking past the end of the stream does not modify the underlying
    ///   stream, but the next write may cause the previous data to be filled in
    ///   with a bit pattern.
    fn seek(&mut self, pos: i64, style: SeekStyle) -> IoResult<()>;
}

/// A listener is a value that can consume itself to start listening for
/// connections.
///
/// Doing so produces some sort of Acceptor.
pub trait Listener<T, A: Acceptor<T>> {
    /// Spin up the listener and start queuing incoming connections
    ///
    /// # Error
    ///
    /// Returns `Err` if this listener could not be bound to listen for
    /// connections. In all cases, this listener is consumed.
    fn listen(self) -> IoResult<A>;
}

/// An acceptor is a value that presents incoming connections
pub trait Acceptor<T> {
    /// Wait for and accept an incoming connection
    ///
    /// # Error
    ///
    /// Returns `Err` if an I/O error is encountered.
    fn accept(&mut self) -> IoResult<T>;

    /// Create an iterator over incoming connection attempts.
    ///
    /// Note that I/O errors will be yielded by the iterator itself.
    fn incoming<'r>(&'r mut self) -> IncomingConnections<'r, Self> { unimplemented!() }
}

/// An infinite iterator over incoming connection attempts.
/// Calling `next` will block the task until a connection is attempted.
///
/// Since connection attempts can continue forever, this iterator always returns
/// `Some`. The `Some` contains the `IoResult` representing whether the
/// connection attempt was successful.  A successful connection will be wrapped
/// in `Ok`. A failed connection is represented as an `Err`.
pub struct IncomingConnections<'a, A:'a> {
    inc: &'a mut A,
}

impl<'a, T, A: Acceptor<T>> Iterator<IoResult<T>> for IncomingConnections<'a, A> {
    fn next(&mut self) -> Option<IoResult<T>> { unimplemented!() }
}

/// Creates a standard error for a commonly used flavor of error. The `detail`
/// field of the returned error will always be `None`.
///
/// # Example
///
/// ```
/// use std::io;
///
/// let eof = io::standard_error(io::EndOfFile);
/// let einval = io::standard_error(io::InvalidInput);
/// ```
pub fn standard_error(kind: IoErrorKind) -> IoError { unimplemented!() }

/// A mode specifies how a file should be opened or created. These modes are
/// passed to `File::open_mode` and are used to control where the file is
/// positioned when it is initially opened.
pub enum FileMode {
    /// Opens a file positioned at the beginning.
    Open,
    /// Opens a file positioned at EOF.
    Append,
    /// Opens a file, truncating it if it already exists.
    Truncate,
}

/// Access permissions with which the file should be opened. `File`s
/// opened with `Read` will return an error if written to.
pub enum FileAccess {
    /// Read-only access, requests to write will result in an error
    Read,
    /// Write-only access, requests to read will result in an error
    Write,
    /// Read-write access, no requests are denied by default
    ReadWrite,
}

/// Different kinds of files which can be identified by a call to stat
#[deriving(PartialEq, Show, Hash, Clone)]
pub enum FileType {
    /// This is a normal file, corresponding to `S_IFREG`
    TypeFile,

    /// This file is a directory, corresponding to `S_IFDIR`
    TypeDirectory,

    /// This file is a named pipe, corresponding to `S_IFIFO`
    TypeNamedPipe,

    /// This file is a block device, corresponding to `S_IFBLK`
    TypeBlockSpecial,

    /// This file is a symbolic link to another file, corresponding to `S_IFLNK`
    TypeSymlink,

    /// The type of this file is not recognized as one of the other categories
    TypeUnknown,
}

/// A structure used to describe metadata information about a file. This
/// structure is created through the `stat` method on a `Path`.
///
/// # Example
///
/// ```
/// # use std::io::fs::PathExtensions;
/// # fn main() {}
/// # fn foo() {
/// let info = match Path::new("foo.txt").stat() {
///     Ok(stat) => stat,
///     Err(e) => panic!("couldn't read foo.txt: {}", e),
/// };
///
/// println!("byte size: {}", info.size);
/// # }
/// ```
#[deriving(Hash)]
pub struct FileStat {
    /// The size of the file, in bytes
    pub size: u64,
    /// The kind of file this path points to (directory, file, pipe, etc.)
    pub kind: FileType,
    /// The file permissions currently on the file
    pub perm: FilePermission,

    // FIXME(#10301): These time fields are pretty useless without an actual
    //                time representation, what are the milliseconds relative
    //                to?

    /// The time that the file was created at, in platform-dependent
    /// milliseconds
    pub created: u64,
    /// The time that this file was last modified, in platform-dependent
    /// milliseconds
    pub modified: u64,
    /// The time that this file was last accessed, in platform-dependent
    /// milliseconds
    pub accessed: u64,

    /// Information returned by stat() which is not guaranteed to be
    /// platform-independent. This information may be useful on some platforms,
    /// but it may have different meanings or no meaning at all on other
    /// platforms.
    ///
    /// Usage of this field is discouraged, but if access is desired then the
    /// fields are located here.
    #[unstable]
    pub unstable: UnstableFileStat,
}

/// This structure represents all of the possible information which can be
/// returned from a `stat` syscall which is not contained in the `FileStat`
/// structure. This information is not necessarily platform independent, and may
/// have different meanings or no meaning at all on some platforms.
#[unstable]
#[deriving(Hash)]
pub struct UnstableFileStat {
    /// The ID of the device containing the file.
    pub device: u64,
    /// The file serial number.
    pub inode: u64,
    /// The device ID.
    pub rdev: u64,
    /// The number of hard links to this file.
    pub nlink: u64,
    /// The user ID of the file.
    pub uid: u64,
    /// The group ID of the file.
    pub gid: u64,
    /// The optimal block size for I/O.
    pub blksize: u64,
    /// The blocks allocated for this file.
    pub blocks: u64,
    /// User-defined flags for the file.
    pub flags: u64,
    /// The file generation number.
    pub gen: u64,
}

bitflags! {
    #[doc = "A set of permissions for a file or directory is represented"]
    #[doc = "by a set of flags which are or'd together."]
    flags FilePermission: u32 {
        const USER_READ     = 0o400,
        const USER_WRITE    = 0o200,
        const USER_EXECUTE  = 0o100,
        const GROUP_READ    = 0o040,
        const GROUP_WRITE   = 0o020,
        const GROUP_EXECUTE = 0o010,
        const OTHER_READ    = 0o004,
        const OTHER_WRITE   = 0o002,
        const OTHER_EXECUTE = 0o001,

        const USER_RWX  = USER_READ.bits | USER_WRITE.bits | USER_EXECUTE.bits,
        const GROUP_RWX = GROUP_READ.bits | GROUP_WRITE.bits | GROUP_EXECUTE.bits,
        const OTHER_RWX = OTHER_READ.bits | OTHER_WRITE.bits | OTHER_EXECUTE.bits,

        #[doc = "Permissions for user owned files, equivalent to 0644 on"]
        #[doc = "unix-like systems."]
        const USER_FILE = USER_READ.bits | USER_WRITE.bits | GROUP_READ.bits | OTHER_READ.bits,

        #[doc = "Permissions for user owned directories, equivalent to 0755 on"]
        #[doc = "unix-like systems."]
        const USER_DIR  = USER_RWX.bits | GROUP_READ.bits | GROUP_EXECUTE.bits |
                   OTHER_READ.bits | OTHER_EXECUTE.bits,

        #[doc = "Permissions for user owned executables, equivalent to 0755"]
        #[doc = "on unix-like systems."]
        const USER_EXEC = USER_DIR.bits,

        #[doc = "All possible permissions enabled."]
        const ALL_PERMISSIONS = USER_RWX.bits | GROUP_RWX.bits | OTHER_RWX.bits,

        // Deprecated names
        #[allow(non_upper_case_globals)]
        #[deprecated = "use USER_READ instead"]
        const UserRead     = USER_READ.bits,
        #[allow(non_upper_case_globals)]
        #[deprecated = "use USER_WRITE instead"]
        const UserWrite    = USER_WRITE.bits,
        #[allow(non_upper_case_globals)]
        #[deprecated = "use USER_EXECUTE instead"]
        const UserExecute  = USER_EXECUTE.bits,
        #[allow(non_upper_case_globals)]
        #[deprecated = "use GROUP_READ instead"]
        const GroupRead    = GROUP_READ.bits,
        #[allow(non_upper_case_globals)]
        #[deprecated = "use GROUP_WRITE instead"]
        const GroupWrite   = GROUP_WRITE.bits,
        #[allow(non_upper_case_globals)]
        #[deprecated = "use GROUP_EXECUTE instead"]
        const GroupExecute = GROUP_EXECUTE.bits,
        #[allow(non_upper_case_globals)]
        #[deprecated = "use OTHER_READ instead"]
        const OtherRead    = OTHER_READ.bits,
        #[allow(non_upper_case_globals)]
        #[deprecated = "use OTHER_WRITE instead"]
        const OtherWrite   = OTHER_WRITE.bits,
        #[allow(non_upper_case_globals)]
        #[deprecated = "use OTHER_EXECUTE instead"]
        const OtherExecute = OTHER_EXECUTE.bits,

        #[allow(non_upper_case_globals)]
        #[deprecated = "use USER_RWX instead"]
        const UserRWX  = USER_RWX.bits,
        #[allow(non_upper_case_globals)]
        #[deprecated = "use GROUP_RWX instead"]
        const GroupRWX = GROUP_RWX.bits,
        #[allow(non_upper_case_globals)]
        #[deprecated = "use OTHER_RWX instead"]
        const OtherRWX = OTHER_RWX.bits,

        #[doc = "Deprecated: use `USER_FILE` instead."]
        #[allow(non_upper_case_globals)]
        #[deprecated = "use USER_FILE instead"]
        const UserFile = USER_FILE.bits,

        #[doc = "Deprecated: use `USER_DIR` instead."]
        #[allow(non_upper_case_globals)]
        #[deprecated = "use USER_DIR instead"]
        const UserDir  = USER_DIR.bits,
        #[doc = "Deprecated: use `USER_EXEC` instead."]
        #[allow(non_upper_case_globals)]
        #[deprecated = "use USER_EXEC instead"]
        const UserExec = USER_EXEC.bits,

        #[doc = "Deprecated: use `ALL_PERMISSIONS` instead"]
        #[allow(non_upper_case_globals)]
        #[deprecated = "use ALL_PERMISSIONS instead"]
        const AllPermissions = ALL_PERMISSIONS.bits,
    }
}

impl Default for FilePermission {
    #[inline]
    fn default() -> FilePermission { unimplemented!() }
}

impl fmt::Show for FilePermission {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result { unimplemented!() }
}
