// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

/*! Synchronous I/O

This module defines the Rust interface for synchronous I/O.
It models byte-oriented input and output with the Reader and Writer traits.
Types that implement both `Reader` and `Writer` are called 'streams',
and automatically implement the `Stream` trait.
Implementations are provided for common I/O streams like
file, TCP, UDP, Unix domain sockets.
Readers and Writers may be composed to add capabilities like string
parsing, encoding, and compression.

# Examples

Some examples of obvious things you might want to do

* Read lines from stdin

    ```rust
    use std::io::buffered::BufferedReader;
    use std::io::stdin;

    # let _g = ::std::io::ignore_io_error();
    let mut stdin = BufferedReader::new(stdin());
    for line in stdin.lines() {
        print(line);
    }
    ```

* Read a complete file

    ```rust
    use std::io::File;

    # let _g = ::std::io::ignore_io_error();
    let contents = File::open(&Path::new("message.txt")).read_to_end();
    ```

* Write a line to a file

    ```rust
    use std::io::File;

    # let _g = ::std::io::ignore_io_error();
    let mut file = File::create(&Path::new("message.txt"));
    file.write(bytes!("hello, file!\n"));
    # drop(file);
    # ::std::io::fs::unlink(&Path::new("message.txt"));
    ```

* Iterate over the lines of a file

    ```rust
    use std::io::buffered::BufferedReader;
    use std::io::File;

    # let _g = ::std::io::ignore_io_error();
    let path = Path::new("message.txt");
    let mut file = BufferedReader::new(File::open(&path));
    for line in file.lines() {
        print(line);
    }
    ```

* Pull the lines of a file into a vector of strings

    ```rust
    use std::io::buffered::BufferedReader;
    use std::io::File;

    # let _g = ::std::io::ignore_io_error();
    let path = Path::new("message.txt");
    let mut file = BufferedReader::new(File::open(&path));
    let lines: ~[~str] = file.lines().collect();
    ```

* Make an simple HTTP request
  XXX This needs more improvement: TcpStream constructor taking &str,
  `write_str` and `write_line` methods.

    ```rust,should_fail
    use std::io::net::ip::SocketAddr;
    use std::io::net::tcp::TcpStream;

    # let _g = ::std::io::ignore_io_error();
    let addr = from_str::<SocketAddr>("127.0.0.1:8080").unwrap();
    let mut socket = TcpStream::connect(addr).unwrap();
    socket.write(bytes!("GET / HTTP/1.0\n\n"));
    let response = socket.read_to_end();
    ```

* Connect based on URL? Requires thinking about where the URL type lives
  and how to make protocol handlers extensible, e.g. the "tcp" protocol
  yields a `TcpStream`.
  XXX this is not implemented now.

    ```rust
    // connect("tcp://localhost:8080");
    ```

# Terms

* Reader - An I/O source, reads bytes into a buffer
* Writer - An I/O sink, writes bytes from a buffer
* Stream - Typical I/O sources like files and sockets are both Readers and Writers,
  and are collectively referred to a `streams`.
* Decorator - A Reader or Writer that composes with others to add additional capabilities
  such as encoding or decoding

# Blocking and synchrony

When discussing I/O you often hear the terms 'synchronous' and
'asynchronous', along with 'blocking' and 'non-blocking' compared and
contrasted. A synchronous I/O interface performs each I/O operation to
completion before proceeding to the next. Synchronous interfaces are
usually used in imperative style as a sequence of commands. An
asynchronous interface allows multiple I/O requests to be issued
simultaneously, without waiting for each to complete before proceeding
to the next.

Asynchronous interfaces are used to achieve 'non-blocking' I/O. In
traditional single-threaded systems, performing a synchronous I/O
operation means that the program stops all activity (it 'blocks')
until the I/O is complete. Blocking is bad for performance when
there are other computations that could be done.

Asynchronous interfaces are most often associated with the callback
(continuation-passing) style popularised by node.js. Such systems rely
on all computations being run inside an event loop which maintains a
list of all pending I/O events; when one completes the registered
callback is run and the code that made the I/O request continues.
Such interfaces achieve non-blocking at the expense of being more
difficult to reason about.

Rust's I/O interface is synchronous - easy to read - and non-blocking by default.

Remember that Rust tasks are 'green threads', lightweight threads that
are multiplexed onto a single operating system thread. If that system
thread blocks then no other task may proceed. Rust tasks are
relatively cheap to create, so as long as other tasks are free to
execute then non-blocking code may be written by simply creating a new
task.

When discussing blocking in regards to Rust's I/O model, we are
concerned with whether performing I/O blocks other Rust tasks from
proceeding. In other words, when a task calls `read`, it must then
wait (or 'sleep', or 'block') until the call to `read` is complete.
During this time, other tasks may or may not be executed, depending on
how `read` is implemented.


Rust's default I/O implementation is non-blocking; by cooperating
directly with the task scheduler it arranges to never block progress
of *other* tasks. Under the hood, Rust uses asynchronous I/O via a
per-scheduler (and hence per-thread) event loop. Synchronous I/O
requests are implemented by descheduling the running task and
performing an asynchronous request; the task is only resumed once the
asynchronous request completes.

# Error Handling

I/O is an area where nearly every operation can result in unexpected
errors. It should allow errors to be handled efficiently.
It needs to be convenient to use I/O when you don't care
about dealing with specific errors.

Rust's I/O employs a combination of techniques to reduce boilerplate
while still providing feedback about errors. The basic strategy:

* Errors are fatal by default, resulting in task failure
* Errors raise the `io_error` condition which provides an opportunity to inspect
  an IoError object containing details.
* Return values must have a sensible null or zero value which is returned
  if a condition is handled successfully. This may be an `Option`, an empty
  vector, or other designated error value.
* Common traits are implemented for `Option`, e.g. `impl<R: Reader> Reader for Option<R>`,
  so that nullable values do not have to be 'unwrapped' before use.

These features combine in the API to allow for expressions like
`File::new("diary.txt").write_line("met a girl")` without having to
worry about whether "diary.txt" exists or whether the write
succeeds. As written, if either `new` or `write_line` encounters
an error the task will fail.

If you wanted to handle the error though you might write

    let mut error = None;
    do io_error::cond(|e: IoError| {
        error = Some(e);
    }).in {
        File::new("diary.txt").write_line("met a girl");
    }

    if error.is_some() {
        println("failed to write my diary");
    }

XXX: Need better condition handling syntax

In this case the condition handler will have the opportunity to
inspect the IoError raised by either the call to `new` or the call to
`write_line`, but then execution will continue.

So what actually happens if `new` encounters an error? To understand
that it's important to know that what `new` returns is not a `File`
but an `Option<File>`.  If the file does not open, and the condition
is handled, then `new` will simply return `None`. Because there is an
implementation of `Writer` (the trait required ultimately required for
types to implement `write_line`) there is no need to inspect or unwrap
the `Option<File>` and we simply call `write_line` on it.  If `new`
returned a `None` then the followup call to `write_line` will also
raise an error.

## Concerns about this strategy

This structure will encourage a programming style that is prone
to errors similar to null pointer dereferences.
In particular code written to ignore errors and expect conditions to be unhandled
will start passing around null or zero objects when wrapped in a condition handler.

* XXX: How should we use condition handlers that return values?
* XXX: Should EOF raise default conditions when EOF is not an error?

# Issues with i/o scheduler affinity, work stealing, task pinning

# Resource management

* `close` vs. RAII

# Paths, URLs and overloaded constructors



# Scope

In scope for core

* Url?

Some I/O things don't belong in core

  - url
  - net - `fn connect`
    - http
  - flate

Out of scope

* Async I/O. We'll probably want it eventually


# XXX Questions and issues

* Should default constructors take `Path` or `&str`? `Path` makes simple cases verbose.
  Overloading would be nice.
* Add overloading for Path and &str and Url &str
* stdin/err/out
* print, println, etc.
* fsync
* relationship with filesystem querying, Directory, File types etc.
* Rename Reader/Writer to ByteReader/Writer, make Reader/Writer generic?
* Can Port and Chan be implementations of a generic Reader<T>/Writer<T>?
* Trait for things that are both readers and writers, Stream?
* How to handle newline conversion
* String conversion
* open vs. connect for generic stream opening
* Do we need `close` at all? dtors might be good enough
* How does I/O relate to the Iterator trait?
* std::base64 filters
* Using conditions is a big unknown since we don't have much experience with them
* Too many uses of OtherIoError

*/

#[allow(missing_doc)];

use cast;
use condition::Guard;
use container::Container;
use int;
use iter::Iterator;
use option::{Option, Some, None};
use path::Path;
use result::{Ok, Err, Result};
use str;
use str::{StrSlice, OwnedStr};
use to_str::ToStr;
use uint;
use unstable::finally::Finally;
use vec::{OwnedVector, MutableVector, ImmutableVector, OwnedCopyableVector};
use vec;

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
pub use self::process::Process;

/// Various utility functions useful for writing I/O tests
pub mod test;

/// Synchronous, non-blocking filesystem operations.
pub mod fs;

/// Synchronous, in-memory I/O.
pub mod pipe;

/// Child process management.
pub mod process;

/// Synchronous, non-blocking network I/O.
pub mod net;

/// Readers and Writers for memory buffers and strings.
pub mod mem;

/// Non-blocking access to stdin, stdout, stderr
pub mod stdio;

/// Implementations for Option
mod option;

/// Basic stream compression. XXX: Belongs with other flate code
pub mod flate;

/// Extension traits
pub mod extensions;

/// Basic Timer
pub mod timer;

/// Buffered I/O wrappers
pub mod buffered;

/// Signal handling
pub mod signal;

/// Utility implementations of Reader and Writer
pub mod util;

/// Adapatation of Chan/Port types to a Writer/Reader type.
pub mod comm_adapters;

/// The default buffer size for various I/O operations
static DEFAULT_BUF_SIZE: uint = 1024 * 64;

/// The type passed to I/O condition handlers to indicate error
///
/// # XXX
///
/// Is something like this sufficient? It's kind of archaic
pub struct IoError {
    kind: IoErrorKind,
    desc: &'static str,
    detail: Option<~str>
}

// FIXME: #8242 implementing manually because deriving doesn't work for some reason
impl ToStr for IoError {
    fn to_str(&self) -> ~str {
        let mut s = ~"IoError { kind: ";
        s.push_str(self.kind.to_str());
        s.push_str(", desc: ");
        s.push_str(self.desc);
        s.push_str(", detail: ");
        s.push_str(self.detail.to_str());
        s.push_str(" }");
        s
    }
}

#[deriving(Eq)]
pub enum IoErrorKind {
    PreviousIoError,
    OtherIoError,
    EndOfFile,
    FileNotFound,
    PermissionDenied,
    ConnectionFailed,
    Closed,
    ConnectionRefused,
    ConnectionReset,
    ConnectionAborted,
    NotConnected,
    BrokenPipe,
    PathAlreadyExists,
    PathDoesntExist,
    MismatchedFileTypeForOperation,
    ResourceUnavailable,
    IoUnavailable,
    InvalidInput,
}

// FIXME: #8242 implementing manually because deriving doesn't work for some reason
impl ToStr for IoErrorKind {
    fn to_str(&self) -> ~str {
        match *self {
            PreviousIoError => ~"PreviousIoError",
            OtherIoError => ~"OtherIoError",
            EndOfFile => ~"EndOfFile",
            FileNotFound => ~"FileNotFound",
            PermissionDenied => ~"PermissionDenied",
            ConnectionFailed => ~"ConnectionFailed",
            Closed => ~"Closed",
            ConnectionRefused => ~"ConnectionRefused",
            ConnectionReset => ~"ConnectionReset",
            NotConnected => ~"NotConnected",
            BrokenPipe => ~"BrokenPipe",
            PathAlreadyExists => ~"PathAlreadyExists",
            PathDoesntExist => ~"PathDoesntExist",
            MismatchedFileTypeForOperation => ~"MismatchedFileTypeForOperation",
            IoUnavailable => ~"IoUnavailable",
            ResourceUnavailable => ~"ResourceUnavailable",
            ConnectionAborted => ~"ConnectionAborted",
            InvalidInput => ~"InvalidInput",
        }
    }
}

// XXX: Can't put doc comments on macros
// Raised by `I/O` operations on error.
condition! {
    pub io_error: IoError -> ();
}

/// Helper for wrapper calls where you want to
/// ignore any io_errors that might be raised
pub fn ignore_io_error() -> Guard<'static,IoError,()> {
    io_error::cond.trap(|_| {
        // just swallow the error.. downstream users
        // who can make a decision based on a None result
        // won't care
    }).guard()
}

/// Helper for catching an I/O error and wrapping it in a Result object. The
/// return result will be the last I/O error that happened or the result of the
/// closure if no error occurred.
pub fn result<T>(cb: || -> T) -> Result<T, IoError> {
    let mut err = None;
    let ret = io_error::cond.trap(|e| {
        if err.is_none() {
            err = Some(e);
        }
    }).inside(cb);
    match err {
        Some(e) => Err(e),
        None => Ok(ret),
    }
}

pub trait Reader {

    // Only two methods which need to get implemented for this trait

    /// Read bytes, up to the length of `buf` and place them in `buf`.
    /// Returns the number of bytes read. The number of bytes read my
    /// be less than the number requested, even 0. Returns `None` on EOF.
    ///
    /// # Failure
    ///
    /// Raises the `io_error` condition on error. If the condition
    /// is handled then no guarantee is made about the number of bytes
    /// read and the contents of `buf`. If the condition is handled
    /// returns `None` (XXX see below).
    ///
    /// # XXX
    ///
    /// * Should raise_default error on eof?
    /// * If the condition is handled it should still return the bytes read,
    ///   in which case there's no need to return Option - but then you *have*
    ///   to install a handler to detect eof.
    ///
    /// This doesn't take a `len` argument like the old `read`.
    /// Will people often need to slice their vectors to call this
    /// and will that be annoying?
    /// Is it actually possible for 0 bytes to be read successfully?
    fn read(&mut self, buf: &mut [u8]) -> Option<uint>;

    /// Return whether the Reader has reached the end of the stream.
    ///
    /// # Example
    ///
    ///     let mut reader = BufferedReader::new(File::open(&Path::new("foo.txt")));
    ///     for line in reader.lines() {
    ///         println(line);
    ///     }
    ///
    /// # Failure
    ///
    /// Returns `true` on failure.
    fn eof(&mut self) -> bool;

    // Convenient helper methods based on the above methods

    /// Reads a single byte. Returns `None` on EOF.
    ///
    /// # Failure
    ///
    /// Raises the same conditions as the `read` method. Returns
    /// `None` if the condition is handled.
    fn read_byte(&mut self) -> Option<u8> {
        let mut buf = [0];
        match self.read(buf) {
            Some(0) => {
                debug!("read 0 bytes. trying again");
                self.read_byte()
            }
            Some(1) => Some(buf[0]),
            Some(_) => unreachable!(),
            None => None
        }
    }

    /// Reads `len` bytes and appends them to a vector.
    ///
    /// May push fewer than the requested number of bytes on error
    /// or EOF. Returns true on success, false on EOF or error.
    ///
    /// # Failure
    ///
    /// Raises the same conditions as `read`. Additionally raises `io_error`
    /// on EOF. If `io_error` is handled then `push_bytes` may push less
    /// than the requested number of bytes.
    fn push_bytes(&mut self, buf: &mut ~[u8], len: uint) {
        unsafe {
            let start_len = buf.len();
            let mut total_read = 0;

            buf.reserve_additional(len);
            buf.set_len(start_len + len);

            (|| {
                while total_read < len {
                    let len = buf.len();
                    let slice = buf.mut_slice(start_len + total_read, len);
                    match self.read(slice) {
                        Some(nread) => {
                            total_read += nread;
                        }
                        None => {
                            io_error::cond.raise(standard_error(EndOfFile));
                            break;
                        }
                    }
                }
            }).finally(|| buf.set_len(start_len + total_read))
        }
    }

    /// Reads `len` bytes and gives you back a new vector of length `len`
    ///
    /// # Failure
    ///
    /// Raises the same conditions as `read`. Additionally raises `io_error`
    /// on EOF. If `io_error` is handled then the returned vector may
    /// contain less than the requested number of bytes.
    fn read_bytes(&mut self, len: uint) -> ~[u8] {
        let mut buf = vec::with_capacity(len);
        self.push_bytes(&mut buf, len);
        return buf;
    }

    /// Reads all remaining bytes from the stream.
    ///
    /// # Failure
    ///
    /// Raises the same conditions as the `read` method except for
    /// `EndOfFile` which is swallowed.
    fn read_to_end(&mut self) -> ~[u8] {
        let mut buf = vec::with_capacity(DEFAULT_BUF_SIZE);
        let mut keep_reading = true;
        io_error::cond.trap(|e| {
            if e.kind == EndOfFile {
                keep_reading = false;
            } else {
                io_error::cond.raise(e)
            }
        }).inside(|| {
            while keep_reading {
                self.push_bytes(&mut buf, DEFAULT_BUF_SIZE)
            }
        });
        return buf;
    }

    /// Reads all of the remaining bytes of this stream, interpreting them as a
    /// UTF-8 encoded stream. The corresponding string is returned.
    ///
    /// # Failure
    ///
    /// This function will raise all the same conditions as the `read` method,
    /// along with raising a condition if the input is not valid UTF-8.
    fn read_to_str(&mut self) -> ~str {
        match str::from_utf8_owned_opt(self.read_to_end()) {
            Some(s) => s,
            None => {
                io_error::cond.raise(standard_error(InvalidInput));
                ~""
            }
        }
    }

    /// Create an iterator that reads a single byte on
    /// each iteration, until EOF.
    ///
    /// # Failure
    ///
    /// Raises the same conditions as the `read` method, for
    /// each call to its `.next()` method.
    /// Ends the iteration if the condition is handled.
    fn bytes<'r>(&'r mut self) -> extensions::ByteIterator<'r, Self> {
        extensions::ByteIterator::new(self)
    }

    // Byte conversion helpers

    /// Reads `n` little-endian unsigned integer bytes.
    ///
    /// `n` must be between 1 and 8, inclusive.
    fn read_le_uint_n(&mut self, nbytes: uint) -> u64 {
        assert!(nbytes > 0 && nbytes <= 8);

        let mut val = 0u64;
        let mut pos = 0;
        let mut i = nbytes;
        while i > 0 {
            val += (self.read_u8() as u64) << pos;
            pos += 8;
            i -= 1;
        }
        val
    }

    /// Reads `n` little-endian signed integer bytes.
    ///
    /// `n` must be between 1 and 8, inclusive.
    fn read_le_int_n(&mut self, nbytes: uint) -> i64 {
        extend_sign(self.read_le_uint_n(nbytes), nbytes)
    }

    /// Reads `n` big-endian unsigned integer bytes.
    ///
    /// `n` must be between 1 and 8, inclusive.
    fn read_be_uint_n(&mut self, nbytes: uint) -> u64 {
        assert!(nbytes > 0 && nbytes <= 8);

        let mut val = 0u64;
        let mut i = nbytes;
        while i > 0 {
            i -= 1;
            val += (self.read_u8() as u64) << i * 8;
        }
        val
    }

    /// Reads `n` big-endian signed integer bytes.
    ///
    /// `n` must be between 1 and 8, inclusive.
    fn read_be_int_n(&mut self, nbytes: uint) -> i64 {
        extend_sign(self.read_be_uint_n(nbytes), nbytes)
    }

    /// Reads a little-endian unsigned integer.
    ///
    /// The number of bytes returned is system-dependant.
    fn read_le_uint(&mut self) -> uint {
        self.read_le_uint_n(uint::bytes) as uint
    }

    /// Reads a little-endian integer.
    ///
    /// The number of bytes returned is system-dependant.
    fn read_le_int(&mut self) -> int {
        self.read_le_int_n(int::bytes) as int
    }

    /// Reads a big-endian unsigned integer.
    ///
    /// The number of bytes returned is system-dependant.
    fn read_be_uint(&mut self) -> uint {
        self.read_be_uint_n(uint::bytes) as uint
    }

    /// Reads a big-endian integer.
    ///
    /// The number of bytes returned is system-dependant.
    fn read_be_int(&mut self) -> int {
        self.read_be_int_n(int::bytes) as int
    }

    /// Reads a big-endian `u64`.
    ///
    /// `u64`s are 8 bytes long.
    fn read_be_u64(&mut self) -> u64 {
        self.read_be_uint_n(8) as u64
    }

    /// Reads a big-endian `u32`.
    ///
    /// `u32`s are 4 bytes long.
    fn read_be_u32(&mut self) -> u32 {
        self.read_be_uint_n(4) as u32
    }

    /// Reads a big-endian `u16`.
    ///
    /// `u16`s are 2 bytes long.
    fn read_be_u16(&mut self) -> u16 {
        self.read_be_uint_n(2) as u16
    }

    /// Reads a big-endian `i64`.
    ///
    /// `i64`s are 8 bytes long.
    fn read_be_i64(&mut self) -> i64 {
        self.read_be_int_n(8) as i64
    }

    /// Reads a big-endian `i32`.
    ///
    /// `i32`s are 4 bytes long.
    fn read_be_i32(&mut self) -> i32 {
        self.read_be_int_n(4) as i32
    }

    /// Reads a big-endian `i16`.
    ///
    /// `i16`s are 2 bytes long.
    fn read_be_i16(&mut self) -> i16 {
        self.read_be_int_n(2) as i16
    }

    /// Reads a big-endian `f64`.
    ///
    /// `f64`s are 8 byte, IEEE754 double-precision floating point numbers.
    fn read_be_f64(&mut self) -> f64 {
        unsafe {
            cast::transmute::<u64, f64>(self.read_be_u64())
        }
    }

    /// Reads a big-endian `f32`.
    ///
    /// `f32`s are 4 byte, IEEE754 single-precision floating point numbers.
    fn read_be_f32(&mut self) -> f32 {
        unsafe {
            cast::transmute::<u32, f32>(self.read_be_u32())
        }
    }

    /// Reads a little-endian `u64`.
    ///
    /// `u64`s are 8 bytes long.
    fn read_le_u64(&mut self) -> u64 {
        self.read_le_uint_n(8) as u64
    }

    /// Reads a little-endian `u32`.
    ///
    /// `u32`s are 4 bytes long.
    fn read_le_u32(&mut self) -> u32 {
        self.read_le_uint_n(4) as u32
    }

    /// Reads a little-endian `u16`.
    ///
    /// `u16`s are 2 bytes long.
    fn read_le_u16(&mut self) -> u16 {
        self.read_le_uint_n(2) as u16
    }

    /// Reads a little-endian `i64`.
    ///
    /// `i64`s are 8 bytes long.
    fn read_le_i64(&mut self) -> i64 {
        self.read_le_int_n(8) as i64
    }

    /// Reads a little-endian `i32`.
    ///
    /// `i32`s are 4 bytes long.
    fn read_le_i32(&mut self) -> i32 {
        self.read_le_int_n(4) as i32
    }

    /// Reads a little-endian `i16`.
    ///
    /// `i16`s are 2 bytes long.
    fn read_le_i16(&mut self) -> i16 {
        self.read_le_int_n(2) as i16
    }

    /// Reads a little-endian `f64`.
    ///
    /// `f64`s are 8 byte, IEEE754 double-precision floating point numbers.
    fn read_le_f64(&mut self) -> f64 {
        unsafe {
            cast::transmute::<u64, f64>(self.read_le_u64())
        }
    }

    /// Reads a little-endian `f32`.
    ///
    /// `f32`s are 4 byte, IEEE754 single-precision floating point numbers.
    fn read_le_f32(&mut self) -> f32 {
        unsafe {
            cast::transmute::<u32, f32>(self.read_le_u32())
        }
    }

    /// Read a u8.
    ///
    /// `u8`s are 1 byte.
    fn read_u8(&mut self) -> u8 {
        match self.read_byte() {
            Some(b) => b as u8,
            None => 0
        }
    }

    /// Read an i8.
    ///
    /// `i8`s are 1 byte.
    fn read_i8(&mut self) -> i8 {
        match self.read_byte() {
            Some(b) => b as i8,
            None => 0
        }
    }

}

impl Reader for ~Reader {
    fn read(&mut self, buf: &mut [u8]) -> Option<uint> { self.read(buf) }
    fn eof(&mut self) -> bool { self.eof() }
}

impl<'a> Reader for &'a mut Reader {
    fn read(&mut self, buf: &mut [u8]) -> Option<uint> { self.read(buf) }
    fn eof(&mut self) -> bool { self.eof() }
}

fn extend_sign(val: u64, nbytes: uint) -> i64 {
    let shift = (8 - nbytes) * 8;
    (val << shift) as i64 >> shift
}

pub trait Writer {
    /// Write the given buffer
    ///
    /// # Failure
    ///
    /// Raises the `io_error` condition on error
    fn write(&mut self, buf: &[u8]);

    /// Flush this output stream, ensuring that all intermediately buffered
    /// contents reach their destination.
    ///
    /// This is by default a no-op and implementers of the `Writer` trait should
    /// decide whether their stream needs to be buffered or not.
    fn flush(&mut self) {}

    /// Write a rust string into this sink.
    ///
    /// The bytes written will be the UTF-8 encoded version of the input string.
    /// If other encodings are desired, it is recommended to compose this stream
    /// with another performing the conversion, or to use `write` with a
    /// converted byte-array instead.
    fn write_str(&mut self, s: &str) {
        self.write(s.as_bytes());
    }

    /// Writes a string into this sink, and then writes a literal newline (`\n`)
    /// byte afterwards. Note that the writing of the newline is *not* atomic in
    /// the sense that the call to `write` is invoked twice (once with the
    /// string and once with a newline character).
    ///
    /// If other encodings or line ending flavors are desired, it is recommended
    /// that the `write` method is used specifically instead.
    fn write_line(&mut self, s: &str) {
        self.write_str(s);
        self.write(['\n' as u8]);
    }

    /// Write the result of passing n through `int::to_str_bytes`.
    fn write_int(&mut self, n: int) {
        int::to_str_bytes(n, 10u, |bytes| self.write(bytes))
    }

    /// Write the result of passing n through `uint::to_str_bytes`.
    fn write_uint(&mut self, n: uint) {
        uint::to_str_bytes(n, 10u, |bytes| self.write(bytes))
    }

    /// Write a little-endian uint (number of bytes depends on system).
    fn write_le_uint(&mut self, n: uint) {
        extensions::u64_to_le_bytes(n as u64, uint::bytes, |v| self.write(v))
    }

    /// Write a little-endian int (number of bytes depends on system).
    fn write_le_int(&mut self, n: int) {
        extensions::u64_to_le_bytes(n as u64, int::bytes, |v| self.write(v))
    }

    /// Write a big-endian uint (number of bytes depends on system).
    fn write_be_uint(&mut self, n: uint) {
        extensions::u64_to_be_bytes(n as u64, uint::bytes, |v| self.write(v))
    }

    /// Write a big-endian int (number of bytes depends on system).
    fn write_be_int(&mut self, n: int) {
        extensions::u64_to_be_bytes(n as u64, int::bytes, |v| self.write(v))
    }

    /// Write a big-endian u64 (8 bytes).
    fn write_be_u64(&mut self, n: u64) {
        extensions::u64_to_be_bytes(n, 8u, |v| self.write(v))
    }

    /// Write a big-endian u32 (4 bytes).
    fn write_be_u32(&mut self, n: u32) {
        extensions::u64_to_be_bytes(n as u64, 4u, |v| self.write(v))
    }

    /// Write a big-endian u16 (2 bytes).
    fn write_be_u16(&mut self, n: u16) {
        extensions::u64_to_be_bytes(n as u64, 2u, |v| self.write(v))
    }

    /// Write a big-endian i64 (8 bytes).
    fn write_be_i64(&mut self, n: i64) {
        extensions::u64_to_be_bytes(n as u64, 8u, |v| self.write(v))
    }

    /// Write a big-endian i32 (4 bytes).
    fn write_be_i32(&mut self, n: i32) {
        extensions::u64_to_be_bytes(n as u64, 4u, |v| self.write(v))
    }

    /// Write a big-endian i16 (2 bytes).
    fn write_be_i16(&mut self, n: i16) {
        extensions::u64_to_be_bytes(n as u64, 2u, |v| self.write(v))
    }

    /// Write a big-endian IEEE754 double-precision floating-point (8 bytes).
    fn write_be_f64(&mut self, f: f64) {
        unsafe {
            self.write_be_u64(cast::transmute(f))
        }
    }

    /// Write a big-endian IEEE754 single-precision floating-point (4 bytes).
    fn write_be_f32(&mut self, f: f32) {
        unsafe {
            self.write_be_u32(cast::transmute(f))
        }
    }

    /// Write a little-endian u64 (8 bytes).
    fn write_le_u64(&mut self, n: u64) {
        extensions::u64_to_le_bytes(n, 8u, |v| self.write(v))
    }

    /// Write a little-endian u32 (4 bytes).
    fn write_le_u32(&mut self, n: u32) {
        extensions::u64_to_le_bytes(n as u64, 4u, |v| self.write(v))
    }

    /// Write a little-endian u16 (2 bytes).
    fn write_le_u16(&mut self, n: u16) {
        extensions::u64_to_le_bytes(n as u64, 2u, |v| self.write(v))
    }

    /// Write a little-endian i64 (8 bytes).
    fn write_le_i64(&mut self, n: i64) {
        extensions::u64_to_le_bytes(n as u64, 8u, |v| self.write(v))
    }

    /// Write a little-endian i32 (4 bytes).
    fn write_le_i32(&mut self, n: i32) {
        extensions::u64_to_le_bytes(n as u64, 4u, |v| self.write(v))
    }

    /// Write a little-endian i16 (2 bytes).
    fn write_le_i16(&mut self, n: i16) {
        extensions::u64_to_le_bytes(n as u64, 2u, |v| self.write(v))
    }

    /// Write a little-endian IEEE754 double-precision floating-point
    /// (8 bytes).
    fn write_le_f64(&mut self, f: f64) {
        unsafe {
            self.write_le_u64(cast::transmute(f))
        }
    }

    /// Write a little-endian IEEE754 single-precision floating-point
    /// (4 bytes).
    fn write_le_f32(&mut self, f: f32) {
        unsafe {
            self.write_le_u32(cast::transmute(f))
        }
    }

    /// Write a u8 (1 byte).
    fn write_u8(&mut self, n: u8) {
        self.write([n])
    }

    /// Write a i8 (1 byte).
    fn write_i8(&mut self, n: i8) {
        self.write([n as u8])
    }
}

impl Writer for ~Writer {
    fn write(&mut self, buf: &[u8]) { self.write(buf) }
    fn flush(&mut self) { self.flush() }
}

impl<'a> Writer for &'a mut Writer {
    fn write(&mut self, buf: &[u8]) { self.write(buf) }
    fn flush(&mut self) { self.flush() }
}

pub trait Stream: Reader + Writer { }

impl<T: Reader + Writer> Stream for T {}

/// An iterator that reads a line on each iteration,
/// until `.read_line()` returns `None`.
///
/// # Notes about the Iteration Protocol
///
/// The `LineIterator` may yield `None` and thus terminate
/// an iteration, but continue to yield elements if iteration
/// is attempted again.
///
/// # Failure
///
/// Raises the same conditions as the `read` method except for `EndOfFile`
/// which is swallowed.
/// Iteration yields `None` if the condition is handled.
struct LineIterator<'r, T> {
    priv buffer: &'r mut T,
}

impl<'r, T: Buffer> Iterator<~str> for LineIterator<'r, T> {
    fn next(&mut self) -> Option<~str> {
        self.buffer.read_line()
    }
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
    /// # Failure
    ///
    /// This function will raise on the `io_error` condition if a read error is
    /// encountered.
    fn fill<'a>(&'a mut self) -> &'a [u8];

    /// Tells this buffer that `amt` bytes have been consumed from the buffer,
    /// so they should no longer be returned in calls to `fill` or `read`.
    fn consume(&mut self, amt: uint);

    /// Reads the next line of input, interpreted as a sequence of UTF-8
    /// encoded unicode codepoints. If a newline is encountered, then the
    /// newline is contained in the returned string.
    ///
    /// # Failure
    ///
    /// This function will raise on the `io_error` condition (except for
    /// `EndOfFile` which is swallowed) if a read error is encountered.
    /// The task will also fail if sequence of bytes leading up to
    /// the newline character are not valid UTF-8.
    fn read_line(&mut self) -> Option<~str> {
        self.read_until('\n' as u8).map(str::from_utf8_owned)
    }

    /// Create an iterator that reads a line on each iteration until EOF.
    ///
    /// # Failure
    ///
    /// Iterator raises the same conditions as the `read` method
    /// except for `EndOfFile`.
    fn lines<'r>(&'r mut self) -> LineIterator<'r, Self> {
        LineIterator {
            buffer: self,
        }
    }

    /// Reads a sequence of bytes leading up to a specified delimiter. Once the
    /// specified byte is encountered, reading ceases and the bytes up to and
    /// including the delimiter are returned.
    ///
    /// # Failure
    ///
    /// This function will raise on the `io_error` condition if a read error is
    /// encountered, except that `EndOfFile` is swallowed.
    fn read_until(&mut self, byte: u8) -> Option<~[u8]> {
        let mut res = ~[];

        io_error::cond.trap(|e| {
            if e.kind != EndOfFile {
                io_error::cond.raise(e);
            }
        }).inside(|| {
            let mut used;
            loop {
                {
                    let available = self.fill();
                    match available.iter().position(|&b| b == byte) {
                        Some(i) => {
                            res.push_all(available.slice_to(i + 1));
                            used = i + 1;
                            break
                        }
                        None => {
                            res.push_all(available);
                            used = available.len();
                        }
                    }
                }
                if used == 0 {
                    break
                }
                self.consume(used);
            }
            self.consume(used);
        });
        return if res.len() == 0 {None} else {Some(res)};

    }

    /// Reads the next utf8-encoded character from the underlying stream.
    ///
    /// This will return `None` if the following sequence of bytes in the
    /// stream are not a valid utf8-sequence, or if an I/O error is encountered.
    ///
    /// # Failure
    ///
    /// This function will raise on the `io_error` condition if a read error is
    /// encountered.
    fn read_char(&mut self) -> Option<char> {
        let width = {
            let available = self.fill();
            if available.len() == 0 { return None } // read error
            str::utf8_char_width(available[0])
        };
        if width == 0 { return None } // not uf8
        let mut buf = [0, ..4];
        match self.read(buf.mut_slice_to(width)) {
            Some(n) if n == width => {}
            Some(..) | None => return None // read error
        }
        match str::from_utf8_opt(buf.slice_to(width)) {
            Some(s) => Some(s.char_at(0)),
            None => None
        }
    }
}

pub enum SeekStyle {
    /// Seek from the beginning of the stream
    SeekSet,
    /// Seek from the end of the stream
    SeekEnd,
    /// Seek from the current position
    SeekCur,
}

/// # XXX
/// * Are `u64` and `i64` the right choices?
pub trait Seek {
    /// Return position of file cursor in the stream
    fn tell(&self) -> u64;

    /// Seek to an offset in a stream
    ///
    /// A successful seek clears the EOF indicator.
    ///
    /// # XXX
    ///
    /// * What is the behavior when seeking past the end of a stream?
    fn seek(&mut self, pos: i64, style: SeekStyle);
}

/// A listener is a value that can consume itself to start listening for connections.
/// Doing so produces some sort of Acceptor.
pub trait Listener<T, A: Acceptor<T>> {
    /// Spin up the listener and start queuing incoming connections
    ///
    /// # Failure
    ///
    /// Raises `io_error` condition. If the condition is handled,
    /// then `listen` returns `None`.
    fn listen(self) -> Option<A>;
}

/// An acceptor is a value that presents incoming connections
pub trait Acceptor<T> {
    /// Wait for and accept an incoming connection
    ///
    /// # Failure
    /// Raise `io_error` condition. If the condition is handled,
    /// then `accept` returns `None`.
    fn accept(&mut self) -> Option<T>;

    /// Create an iterator over incoming connection attempts
    fn incoming<'r>(&'r mut self) -> IncomingIterator<'r, Self> {
        IncomingIterator { inc: self }
    }
}

/// An infinite iterator over incoming connection attempts.
/// Calling `next` will block the task until a connection is attempted.
///
/// Since connection attempts can continue forever, this iterator always returns Some.
/// The Some contains another Option representing whether the connection attempt was succesful.
/// A successful connection will be wrapped in Some.
/// A failed connection is represented as a None and raises a condition.
struct IncomingIterator<'a, A> {
    priv inc: &'a mut A,
}

impl<'a, T, A: Acceptor<T>> Iterator<Option<T>> for IncomingIterator<'a, A> {
    fn next(&mut self) -> Option<Option<T>> {
        Some(self.inc.accept())
    }
}

/// Common trait for decorator types.
///
/// Provides accessors to get the inner, 'decorated' values. The I/O library
/// uses decorators to add functionality like compression and encryption to I/O
/// streams.
///
/// # XXX
///
/// Is this worth having a trait for? May be overkill
pub trait Decorator<T> {
    /// Destroy the decorator and extract the decorated value
    ///
    /// # XXX
    ///
    /// Because this takes `self' one could never 'undecorate' a Reader/Writer
    /// that has been boxed. Is that ok? This feature is mostly useful for
    /// extracting the buffer from MemWriter
    fn inner(self) -> T;

    /// Take an immutable reference to the decorated value
    fn inner_ref<'a>(&'a self) -> &'a T;

    /// Take a mutable reference to the decorated value
    fn inner_mut_ref<'a>(&'a mut self) -> &'a mut T;
}

pub fn standard_error(kind: IoErrorKind) -> IoError {
    let desc = match kind {
        PreviousIoError => "failing due to previous I/O error",
        EndOfFile => "end of file",
        IoUnavailable => "I/O is unavailable",
        InvalidInput => "invalid input",
        _ => fail!()
    };
    IoError {
        kind: kind,
        desc: desc,
        detail: None,
    }
}

pub fn placeholder_error() -> IoError {
    IoError {
        kind: OtherIoError,
        desc: "Placeholder error. You shouldn't be seeing this",
        detail: None
    }
}

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
/// opened with `Read` will raise an `io_error` condition if written to.
pub enum FileAccess {
    Read,
    Write,
    ReadWrite,
}

/// Different kinds of files which can be identified by a call to stat
#[deriving(Eq)]
pub enum FileType {
    TypeFile,
    TypeDirectory,
    TypeNamedPipe,
    TypeBlockSpecial,
    TypeSymlink,
    TypeUnknown,
}

pub struct FileStat {
    /// The path that this stat structure is describing
    path: Path,
    /// The size of the file, in bytes
    size: u64,
    /// The kind of file this path points to (directory, file, pipe, etc.)
    kind: FileType,
    /// The file permissions currently on the file
    perm: FilePermission,

    // FIXME(#10301): These time fields are pretty useless without an actual
    //                time representation, what are the milliseconds relative
    //                to?

    /// The time that the file was created at, in platform-dependent
    /// milliseconds
    created: u64,
    /// The time that this file was last modified, in platform-dependent
    /// milliseconds
    modified: u64,
    /// The time that this file was last accessed, in platform-dependent
    /// milliseconds
    accessed: u64,

    /// Information returned by stat() which is not guaranteed to be
    /// platform-independent. This information may be useful on some platforms,
    /// but it may have different meanings or no meaning at all on other
    /// platforms.
    ///
    /// Usage of this field is discouraged, but if access is desired then the
    /// fields are located here.
    #[unstable]
    unstable: UnstableFileStat,
}

/// This structure represents all of the possible information which can be
/// returned from a `stat` syscall which is not contained in the `FileStat`
/// structure. This information is not necessarily platform independent, and may
/// have different meanings or no meaning at all on some platforms.
#[unstable]
pub struct UnstableFileStat {
    device: u64,
    inode: u64,
    rdev: u64,
    nlink: u64,
    uid: u64,
    gid: u64,
    blksize: u64,
    blocks: u64,
    flags: u64,
    gen: u64,
}

/// A set of permissions for a file or directory is represented by a set of
/// flags which are or'd together.
pub type FilePermission = u32;

// Each permission bit
pub static UserRead: FilePermission     = 0x100;
pub static UserWrite: FilePermission    = 0x080;
pub static UserExecute: FilePermission  = 0x040;
pub static GroupRead: FilePermission    = 0x020;
pub static GroupWrite: FilePermission   = 0x010;
pub static GroupExecute: FilePermission = 0x008;
pub static OtherRead: FilePermission    = 0x004;
pub static OtherWrite: FilePermission   = 0x002;
pub static OtherExecute: FilePermission = 0x001;

// Common combinations of these bits
pub static UserRWX: FilePermission  = UserRead | UserWrite | UserExecute;
pub static GroupRWX: FilePermission = GroupRead | GroupWrite | GroupExecute;
pub static OtherRWX: FilePermission = OtherRead | OtherWrite | OtherExecute;

/// A set of permissions for user owned files, this is equivalent to 0644 on
/// unix-like systems.
pub static UserFile: FilePermission = UserRead | UserWrite | GroupRead | OtherRead;
/// A set of permissions for user owned directories, this is equivalent to 0755
/// on unix-like systems.
pub static UserDir: FilePermission = UserRWX | GroupRead | GroupExecute |
                                     OtherRead | OtherExecute;
/// A set of permissions for user owned executables, this is equivalent to 0755
/// on unix-like systems.
pub static UserExec: FilePermission = UserDir;

/// A mask for all possible permission bits
pub static AllPermissions: FilePermission = 0x1ff;
