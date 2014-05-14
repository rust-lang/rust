// Copyright 2013-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// FIXME: cover these topics:
//        path, reader, writer, stream, raii (close not needed),
//        stdio, print!, println!, file access, process spawning,
//        error handling


/*! I/O, including files, networking, timers, and processes

`std::io` provides Rust's basic I/O types,
for reading and writing to files, TCP, UDP,
and other types of sockets and pipes,
manipulating the file system, spawning processes and signal handling.

# Examples

Some examples of obvious things you might want to do

* Read lines from stdin

    ```rust
    use std::io;

    for line in io::stdin().lines() {
        print!("{}", line.unwrap());
    }
    ```

* Read a complete file

    ```rust
    use std::io::File;

    let contents = File::open(&Path::new("message.txt")).read_to_end();
    ```

* Write a line to a file

    ```rust
    # #![allow(unused_must_use)]
    use std::io::File;

    let mut file = File::create(&Path::new("message.txt"));
    file.write(bytes!("hello, file!\n"));
    # drop(file);
    # ::std::io::fs::unlink(&Path::new("message.txt"));
    ```

* Iterate over the lines of a file

    ```rust,no_run
    use std::io::BufferedReader;
    use std::io::File;

    let path = Path::new("message.txt");
    let mut file = BufferedReader::new(File::open(&path));
    for line in file.lines() {
        print!("{}", line.unwrap());
    }
    ```

* Pull the lines of a file into a vector of strings

    ```rust,no_run
    use std::io::BufferedReader;
    use std::io::File;

    let path = Path::new("message.txt");
    let mut file = BufferedReader::new(File::open(&path));
    let lines: Vec<~str> = file.lines().map(|x| x.unwrap()).collect();
    ```

* Make a simple TCP client connection and request

    ```rust,should_fail
    # #![allow(unused_must_use)]
    use std::io::net::tcp::TcpStream;

    let mut socket = TcpStream::connect("127.0.0.1", 8080).unwrap();
    socket.write(bytes!("GET / HTTP/1.0\n\n"));
    let response = socket.read_to_end();
    ```

* Make a simple TCP server

    ```rust
    # fn main() { }
    # fn foo() {
    # #![allow(dead_code)]
    use std::io::{TcpListener, TcpStream};
    use std::io::{Acceptor, Listener};

    let listener = TcpListener::bind("127.0.0.1", 80);

    // bind the listener to the specified address
    let mut acceptor = listener.listen();

    fn handle_client(mut stream: TcpStream) {
        // ...
    # &mut stream; // silence unused mutability/variable warning
    }
    // accept connections and process them, spawning a new tasks for each one
    for stream in acceptor.incoming() {
        match stream {
            Err(e) => { /* connection failed */ }
            Ok(stream) => spawn(proc() {
                // connection succeeded
                handle_client(stream)
            })
        }
    }

    // close the socket server
    drop(acceptor);
    # }
    ```


# Error Handling

I/O is an area where nearly every operation can result in unexpected
errors. Errors should be painfully visible when they happen, and handling them
should be easy to work with. It should be convenient to handle specific I/O
errors, and it should also be convenient to not deal with I/O errors.

Rust's I/O employs a combination of techniques to reduce boilerplate
while still providing feedback about errors. The basic strategy:

* All I/O operations return `IoResult<T>` which is equivalent to
  `Result<T, IoError>`. The `Result` type is defined in the `std::result`
  module.
* If the `Result` type goes unused, then the compiler will by default emit a
  warning about the unused result. This is because `Result` has the
  `#[must_use]` attribute.
* Common traits are implemented for `IoResult`, e.g.
  `impl<R: Reader> Reader for IoResult<R>`, so that error values do not have
  to be 'unwrapped' before use.

These features combine in the API to allow for expressions like
`File::create(&Path::new("diary.txt")).write(bytes!("Met a girl.\n"))`
without having to worry about whether "diary.txt" exists or whether
the write succeeds. As written, if either `new` or `write_line`
encounters an error then the result of the entire expression will
be an error.

If you wanted to handle the error though you might write:

```rust
# #![allow(unused_must_use)]
use std::io::File;

match File::create(&Path::new("diary.txt")).write(bytes!("Met a girl.\n")) {
    Ok(()) => (), // succeeded
    Err(e) => println!("failed to write to my diary: {}", e),
}

# ::std::io::fs::unlink(&Path::new("diary.txt"));
```

So what actually happens if `create` encounters an error?
It's important to know that what `new` returns is not a `File`
but an `IoResult<File>`.  If the file does not open, then `new` will simply
return `Err(..)`. Because there is an implementation of `Writer` (the trait
required ultimately required for types to implement `write_line`) there is no
need to inspect or unwrap the `IoResult<File>` and we simply call `write_line`
on it. If `new` returned an `Err(..)` then the followup call to `write_line`
will also return an error.

## `try!`

Explicit pattern matching on `IoResult`s can get quite verbose, especially
when performing many I/O operations. Some examples (like those above) are
alleviated with extra methods implemented on `IoResult`, but others have more
complex interdependencies among each I/O operation.

The `try!` macro from `std::macros` is provided as a method of early-return
inside `Result`-returning functions. It expands to an early-return on `Err`
and otherwise unwraps the contained `Ok` value.

If you wanted to read several `u32`s from a file and return their product:

```rust
use std::io::{File, IoResult};

fn file_product(p: &Path) -> IoResult<u32> {
    let mut f = File::open(p);
    let x1 = try!(f.read_le_u32());
    let x2 = try!(f.read_le_u32());

    Ok(x1 * x2)
}

match file_product(&Path::new("numbers.bin")) {
    Ok(x) => println!("{}", x),
    Err(e) => println!("Failed to read numbers!")
}
```

With `try!` in `file_product`, each `read_le_u32` need not be directly
concerned with error handling; instead its caller is responsible for
responding to errors that may occur while attempting to read the numbers.

*/

#![deny(unused_must_use)]

use char::Char;
use container::Container;
use fmt;
use int;
use iter::Iterator;
use libc;
use mem::transmute;
use ops::{BitOr, BitAnd, Sub, Not};
use option::{Option, Some, None};
use os;
use owned::Box;
use result::{Ok, Err, Result};
use slice::{Vector, MutableVector, ImmutableVector};
use str::{StrSlice, StrAllocating};
use str;
use uint;
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
pub use self::process::{Process, ProcessConfig};
pub use self::tempfile::TempDir;

pub use self::mem::{MemReader, BufReader, MemWriter, BufWriter};
pub use self::buffered::{BufferedReader, BufferedWriter, BufferedStream,
                         LineBufferedWriter};
pub use self::comm_adapters::{ChanReader, ChanWriter};

// this comes first to get the iotest! macro
pub mod test;

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
pub mod signal;
pub mod stdio;
pub mod timer;
pub mod util;

/// The default buffer size for various I/O operations
// libuv recommends 64k buffers to maximize throughput
// https://groups.google.com/forum/#!topic/libuv/oQO1HJAIDdA
static DEFAULT_BUF_SIZE: uint = 1024 * 64;

/// A convenient typedef of the return value of any I/O action.
pub type IoResult<T> = Result<T, IoError>;

/// The type passed to I/O condition handlers to indicate error
///
/// # FIXME
///
/// Is something like this sufficient? It's kind of archaic
#[deriving(Eq, Clone)]
pub struct IoError {
    /// An enumeration which can be matched against for determining the flavor
    /// of error.
    pub kind: IoErrorKind,
    /// A human-readable description about the error
    pub desc: &'static str,
    /// Detailed information about this error, not always available
    pub detail: Option<~str>
}

impl IoError {
    /// Convert an `errno` value into an `IoError`.
    ///
    /// If `detail` is `true`, the `detail` field of the `IoError`
    /// struct is filled with an allocated string describing the error
    /// in more detail, retrieved from the operating system.
    pub fn from_errno(errno: uint, detail: bool) -> IoError {
        #[cfg(windows)]
        fn get_err(errno: i32) -> (IoErrorKind, &'static str) {
            match errno {
                libc::EOF => (EndOfFile, "end of file"),
                libc::ERROR_NO_DATA => (BrokenPipe, "the pipe is being closed"),
                libc::ERROR_FILE_NOT_FOUND => (FileNotFound, "file not found"),
                libc::ERROR_INVALID_NAME => (InvalidInput, "invalid file name"),
                libc::WSAECONNREFUSED => (ConnectionRefused, "connection refused"),
                libc::WSAECONNRESET => (ConnectionReset, "connection reset"),
                libc::WSAEACCES => (PermissionDenied, "permission denied"),
                libc::WSAEWOULDBLOCK => {
                    (ResourceUnavailable, "resource temporarily unavailable")
                }
                libc::WSAENOTCONN => (NotConnected, "not connected"),
                libc::WSAECONNABORTED => (ConnectionAborted, "connection aborted"),
                libc::WSAEADDRNOTAVAIL => (ConnectionRefused, "address not available"),
                libc::WSAEADDRINUSE => (ConnectionRefused, "address in use"),
                libc::ERROR_BROKEN_PIPE => (EndOfFile, "the pipe has ended"),
                libc::ERROR_OPERATION_ABORTED =>
                    (TimedOut, "operation timed out"),

                // libuv maps this error code to EISDIR. we do too. if it is found
                // to be incorrect, we can add in some more machinery to only
                // return this message when ERROR_INVALID_FUNCTION after certain
                // win32 calls.
                libc::ERROR_INVALID_FUNCTION => (InvalidInput,
                                                 "illegal operation on a directory"),

                _ => (OtherIoError, "unknown error")
            }
        }

        #[cfg(not(windows))]
        fn get_err(errno: i32) -> (IoErrorKind, &'static str) {
            // FIXME: this should probably be a bit more descriptive...
            match errno {
                libc::EOF => (EndOfFile, "end of file"),
                libc::ECONNREFUSED => (ConnectionRefused, "connection refused"),
                libc::ECONNRESET => (ConnectionReset, "connection reset"),
                libc::EPERM | libc::EACCES =>
                    (PermissionDenied, "permission denied"),
                libc::EPIPE => (BrokenPipe, "broken pipe"),
                libc::ENOTCONN => (NotConnected, "not connected"),
                libc::ECONNABORTED => (ConnectionAborted, "connection aborted"),
                libc::EADDRNOTAVAIL => (ConnectionRefused, "address not available"),
                libc::EADDRINUSE => (ConnectionRefused, "address in use"),
                libc::ENOENT => (FileNotFound, "no such file or directory"),
                libc::EISDIR => (InvalidInput, "illegal operation on a directory"),

                // These two constants can have the same value on some systems, but
                // different values on others, so we can't use a match clause
                x if x == libc::EAGAIN || x == libc::EWOULDBLOCK =>
                    (ResourceUnavailable, "resource temporarily unavailable"),

                _ => (OtherIoError, "unknown error")
            }
        }

        let (kind, desc) = get_err(errno as i32);
        IoError {
            kind: kind,
            desc: desc,
            detail: if detail {Some(os::error_string(errno))} else {None},
        }
    }

    /// Retrieve the last error to occur as a (detailed) IoError.
    ///
    /// This uses the OS `errno`, and so there should not be any task
    /// descheduling or migration (other than that performed by the
    /// operating system) between the call(s) for which errors are
    /// being checked and the call of this function.
    pub fn last_error() -> IoError {
        IoError::from_errno(os::errno() as uint, true)
    }
}

impl fmt::Show for IoError {
    fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
        try!(fmt.buf.write_str(self.desc));
        match self.detail {
            Some(ref s) => write!(fmt.buf, " ({})", *s),
            None => Ok(())
        }
    }
}

/// A list specifying general categories of I/O error.
#[deriving(Eq, Clone, Show)]
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
    /// The network operation failed because the network connection was cloesd.
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
    /// # Implementaton Note
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
    fn read_at_least(&mut self, min: uint, buf: &mut [u8]) -> IoResult<uint> {
        if min > buf.len() {
            return Err(IoError { detail: Some("the buffer is too short".to_owned()),
                                 ..standard_error(InvalidInput) });
        }
        let mut read = 0;
        while read < min {
            let mut zeroes = 0;
            loop {
                match self.read(buf.mut_slice_from(read)) {
                    Ok(0) => {
                        zeroes += 1;
                        if zeroes >= NO_PROGRESS_LIMIT {
                            return Err(standard_error(NoProgress));
                        }
                    }
                    Ok(n) => {
                        read += n;
                        break;
                    }
                    err@Err(_) => return err
                }
            }
        }
        Ok(read)
    }

    /// Reads a single byte. Returns `Err` on EOF.
    fn read_byte(&mut self) -> IoResult<u8> {
        let mut buf = [0];
        try!(self.read_at_least(1, buf));
        Ok(buf[0])
    }

    /// Reads up to `len` bytes and appends them to a vector.
    /// Returns the number of bytes read. The number of bytes read may be
    /// less than the number requested, even 0. Returns Err on EOF.
    ///
    /// # Error
    ///
    /// If an error occurs during this I/O operation, then it is returned
    /// as `Err(IoError)`. See `read()` for more details.
    fn push(&mut self, len: uint, buf: &mut Vec<u8>) -> IoResult<uint> {
        let start_len = buf.len();
        buf.reserve_additional(len);

        let n = {
            let s = unsafe { slice_vec_capacity(buf, start_len, start_len + len) };
            try!(self.read(s))
        };
        unsafe { buf.set_len(start_len + n) };
        Ok(n)
    }

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
    fn push_at_least(&mut self, min: uint, len: uint, buf: &mut Vec<u8>) -> IoResult<uint> {
        if min > len {
            return Err(IoError { detail: Some("the buffer is too short".to_owned()),
                                 ..standard_error(InvalidInput) });
        }

        let start_len = buf.len();
        buf.reserve_additional(len);

        // we can't just use self.read_at_least(min, slice) because we need to push
        // successful reads onto the vector before any returned errors.

        let mut read = 0;
        while read < min {
            read += {
                let s = unsafe { slice_vec_capacity(buf, start_len + read, start_len + len) };
                try!(self.read_at_least(1, s))
            };
            unsafe { buf.set_len(start_len + read) };
        }
        Ok(read)
    }

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
    fn read_exact(&mut self, len: uint) -> IoResult<Vec<u8>> {
        let mut buf = Vec::with_capacity(len);
        match self.push_at_least(len, len, &mut buf) {
            Ok(_) => Ok(buf),
            Err(e) => Err(e),
        }
    }

    /// Reads all remaining bytes from the stream.
    ///
    /// # Error
    ///
    /// Returns any non-EOF error immediately. Previously read bytes are
    /// discarded when an error is returned.
    ///
    /// When EOF is encountered, all bytes read up to that point are returned.
    fn read_to_end(&mut self) -> IoResult<Vec<u8>> {
        let mut buf = Vec::with_capacity(DEFAULT_BUF_SIZE);
        loop {
            match self.push_at_least(1, DEFAULT_BUF_SIZE, &mut buf) {
                Ok(_) => {}
                Err(ref e) if e.kind == EndOfFile => break,
                Err(e) => return Err(e)
            }
        }
        return Ok(buf);
    }

    /// Reads all of the remaining bytes of this stream, interpreting them as a
    /// UTF-8 encoded stream. The corresponding string is returned.
    ///
    /// # Error
    ///
    /// This function returns all of the same errors as `read_to_end` with an
    /// additional error if the reader's contents are not a valid sequence of
    /// UTF-8 bytes.
    fn read_to_str(&mut self) -> IoResult<~str> {
        self.read_to_end().and_then(|s| {
            match str::from_utf8(s.as_slice()) {
                Some(s) => Ok(s.to_owned()),
                None => Err(standard_error(InvalidInput)),
            }
        })
    }

    /// Create an iterator that reads a single byte on
    /// each iteration, until EOF.
    ///
    /// # Error
    ///
    /// Any error other than `EndOfFile` that is produced by the underlying Reader
    /// is returned by the iterator and should be handled by the caller.
    fn bytes<'r>(&'r mut self) -> extensions::Bytes<'r, Self> {
        extensions::Bytes::new(self)
    }

    // Byte conversion helpers

    /// Reads `n` little-endian unsigned integer bytes.
    ///
    /// `n` must be between 1 and 8, inclusive.
    fn read_le_uint_n(&mut self, nbytes: uint) -> IoResult<u64> {
        assert!(nbytes > 0 && nbytes <= 8);

        let mut val = 0u64;
        let mut pos = 0;
        let mut i = nbytes;
        while i > 0 {
            val += (try!(self.read_u8()) as u64) << pos;
            pos += 8;
            i -= 1;
        }
        Ok(val)
    }

    /// Reads `n` little-endian signed integer bytes.
    ///
    /// `n` must be between 1 and 8, inclusive.
    fn read_le_int_n(&mut self, nbytes: uint) -> IoResult<i64> {
        self.read_le_uint_n(nbytes).map(|i| extend_sign(i, nbytes))
    }

    /// Reads `n` big-endian unsigned integer bytes.
    ///
    /// `n` must be between 1 and 8, inclusive.
    fn read_be_uint_n(&mut self, nbytes: uint) -> IoResult<u64> {
        assert!(nbytes > 0 && nbytes <= 8);

        let mut val = 0u64;
        let mut i = nbytes;
        while i > 0 {
            i -= 1;
            val += (try!(self.read_u8()) as u64) << i * 8;
        }
        Ok(val)
    }

    /// Reads `n` big-endian signed integer bytes.
    ///
    /// `n` must be between 1 and 8, inclusive.
    fn read_be_int_n(&mut self, nbytes: uint) -> IoResult<i64> {
        self.read_be_uint_n(nbytes).map(|i| extend_sign(i, nbytes))
    }

    /// Reads a little-endian unsigned integer.
    ///
    /// The number of bytes returned is system-dependent.
    fn read_le_uint(&mut self) -> IoResult<uint> {
        self.read_le_uint_n(uint::BYTES).map(|i| i as uint)
    }

    /// Reads a little-endian integer.
    ///
    /// The number of bytes returned is system-dependent.
    fn read_le_int(&mut self) -> IoResult<int> {
        self.read_le_int_n(int::BYTES).map(|i| i as int)
    }

    /// Reads a big-endian unsigned integer.
    ///
    /// The number of bytes returned is system-dependent.
    fn read_be_uint(&mut self) -> IoResult<uint> {
        self.read_be_uint_n(uint::BYTES).map(|i| i as uint)
    }

    /// Reads a big-endian integer.
    ///
    /// The number of bytes returned is system-dependent.
    fn read_be_int(&mut self) -> IoResult<int> {
        self.read_be_int_n(int::BYTES).map(|i| i as int)
    }

    /// Reads a big-endian `u64`.
    ///
    /// `u64`s are 8 bytes long.
    fn read_be_u64(&mut self) -> IoResult<u64> {
        self.read_be_uint_n(8)
    }

    /// Reads a big-endian `u32`.
    ///
    /// `u32`s are 4 bytes long.
    fn read_be_u32(&mut self) -> IoResult<u32> {
        self.read_be_uint_n(4).map(|i| i as u32)
    }

    /// Reads a big-endian `u16`.
    ///
    /// `u16`s are 2 bytes long.
    fn read_be_u16(&mut self) -> IoResult<u16> {
        self.read_be_uint_n(2).map(|i| i as u16)
    }

    /// Reads a big-endian `i64`.
    ///
    /// `i64`s are 8 bytes long.
    fn read_be_i64(&mut self) -> IoResult<i64> {
        self.read_be_int_n(8)
    }

    /// Reads a big-endian `i32`.
    ///
    /// `i32`s are 4 bytes long.
    fn read_be_i32(&mut self) -> IoResult<i32> {
        self.read_be_int_n(4).map(|i| i as i32)
    }

    /// Reads a big-endian `i16`.
    ///
    /// `i16`s are 2 bytes long.
    fn read_be_i16(&mut self) -> IoResult<i16> {
        self.read_be_int_n(2).map(|i| i as i16)
    }

    /// Reads a big-endian `f64`.
    ///
    /// `f64`s are 8 byte, IEEE754 double-precision floating point numbers.
    fn read_be_f64(&mut self) -> IoResult<f64> {
        self.read_be_u64().map(|i| unsafe {
            transmute::<u64, f64>(i)
        })
    }

    /// Reads a big-endian `f32`.
    ///
    /// `f32`s are 4 byte, IEEE754 single-precision floating point numbers.
    fn read_be_f32(&mut self) -> IoResult<f32> {
        self.read_be_u32().map(|i| unsafe {
            transmute::<u32, f32>(i)
        })
    }

    /// Reads a little-endian `u64`.
    ///
    /// `u64`s are 8 bytes long.
    fn read_le_u64(&mut self) -> IoResult<u64> {
        self.read_le_uint_n(8)
    }

    /// Reads a little-endian `u32`.
    ///
    /// `u32`s are 4 bytes long.
    fn read_le_u32(&mut self) -> IoResult<u32> {
        self.read_le_uint_n(4).map(|i| i as u32)
    }

    /// Reads a little-endian `u16`.
    ///
    /// `u16`s are 2 bytes long.
    fn read_le_u16(&mut self) -> IoResult<u16> {
        self.read_le_uint_n(2).map(|i| i as u16)
    }

    /// Reads a little-endian `i64`.
    ///
    /// `i64`s are 8 bytes long.
    fn read_le_i64(&mut self) -> IoResult<i64> {
        self.read_le_int_n(8)
    }

    /// Reads a little-endian `i32`.
    ///
    /// `i32`s are 4 bytes long.
    fn read_le_i32(&mut self) -> IoResult<i32> {
        self.read_le_int_n(4).map(|i| i as i32)
    }

    /// Reads a little-endian `i16`.
    ///
    /// `i16`s are 2 bytes long.
    fn read_le_i16(&mut self) -> IoResult<i16> {
        self.read_le_int_n(2).map(|i| i as i16)
    }

    /// Reads a little-endian `f64`.
    ///
    /// `f64`s are 8 byte, IEEE754 double-precision floating point numbers.
    fn read_le_f64(&mut self) -> IoResult<f64> {
        self.read_le_u64().map(|i| unsafe {
            transmute::<u64, f64>(i)
        })
    }

    /// Reads a little-endian `f32`.
    ///
    /// `f32`s are 4 byte, IEEE754 single-precision floating point numbers.
    fn read_le_f32(&mut self) -> IoResult<f32> {
        self.read_le_u32().map(|i| unsafe {
            transmute::<u32, f32>(i)
        })
    }

    /// Read a u8.
    ///
    /// `u8`s are 1 byte.
    fn read_u8(&mut self) -> IoResult<u8> {
        self.read_byte()
    }

    /// Read an i8.
    ///
    /// `i8`s are 1 byte.
    fn read_i8(&mut self) -> IoResult<i8> {
        self.read_byte().map(|i| i as i8)
    }

    /// Creates a wrapper around a mutable reference to the reader.
    ///
    /// This is useful to allow applying adaptors while still
    /// retaining ownership of the original value.
    fn by_ref<'a>(&'a mut self) -> RefReader<'a, Self> {
        RefReader { inner: self }
    }
}

impl Reader for Box<Reader> {
    fn read(&mut self, buf: &mut [u8]) -> IoResult<uint> { self.read(buf) }
}

impl<'a> Reader for &'a mut Reader {
    fn read(&mut self, buf: &mut [u8]) -> IoResult<uint> { self.read(buf) }
}

/// Returns a slice of `v` between `start` and `end`.
///
/// Similar to `slice()` except this function only bounds the sclie on the
/// capacity of `v`, not the length.
///
/// # Failure
///
/// Fails when `start` or `end` point outside the capacity of `v`, or when
/// `start` > `end`.
// Private function here because we aren't sure if we want to expose this as
// API yet. If so, it should be a method on Vec.
unsafe fn slice_vec_capacity<'a, T>(v: &'a mut Vec<T>, start: uint, end: uint) -> &'a mut [T] {
    use raw::Slice;
    use ptr::RawPtr;

    assert!(start <= end);
    assert!(end <= v.capacity());
    transmute(Slice {
        data: v.as_ptr().offset(start as int),
        len: end - start
    })
}

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
pub struct RefReader<'a, R> {
    /// The underlying reader which this is referencing
    inner: &'a mut R
}

impl<'a, R: Reader> Reader for RefReader<'a, R> {
    fn read(&mut self, buf: &mut [u8]) -> IoResult<uint> { self.inner.read(buf) }
}

impl<'a, R: Buffer> Buffer for RefReader<'a, R> {
    fn fill_buf<'a>(&'a mut self) -> IoResult<&'a [u8]> { self.inner.fill_buf() }
    fn consume(&mut self, amt: uint) { self.inner.consume(amt) }
}

fn extend_sign(val: u64, nbytes: uint) -> i64 {
    let shift = (8 - nbytes) * 8;
    (val << shift) as i64 >> shift
}

/// A trait for objects which are byte-oriented streams. Writers are defined by
/// one method, `write`. This function will block until the provided buffer of
/// bytes has been entirely written, and it will return any failurs which occur.
///
/// Another commonly overriden method is the `flush` method for writers such as
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
    fn flush(&mut self) -> IoResult<()> { Ok(()) }

    /// Write a rust string into this sink.
    ///
    /// The bytes written will be the UTF-8 encoded version of the input string.
    /// If other encodings are desired, it is recommended to compose this stream
    /// with another performing the conversion, or to use `write` with a
    /// converted byte-array instead.
    fn write_str(&mut self, s: &str) -> IoResult<()> {
        self.write(s.as_bytes())
    }

    /// Writes a string into this sink, and then writes a literal newline (`\n`)
    /// byte afterwards. Note that the writing of the newline is *not* atomic in
    /// the sense that the call to `write` is invoked twice (once with the
    /// string and once with a newline character).
    ///
    /// If other encodings or line ending flavors are desired, it is recommended
    /// that the `write` method is used specifically instead.
    fn write_line(&mut self, s: &str) -> IoResult<()> {
        self.write_str(s).and_then(|()| self.write(['\n' as u8]))
    }

    /// Write a single char, encoded as UTF-8.
    fn write_char(&mut self, c: char) -> IoResult<()> {
        let mut buf = [0u8, ..4];
        let n = c.encode_utf8(buf.as_mut_slice());
        self.write(buf.slice_to(n))
    }

    /// Write the result of passing n through `int::to_str_bytes`.
    fn write_int(&mut self, n: int) -> IoResult<()> {
        write!(self, "{:d}", n)
    }

    /// Write the result of passing n through `uint::to_str_bytes`.
    fn write_uint(&mut self, n: uint) -> IoResult<()> {
        write!(self, "{:u}", n)
    }

    /// Write a little-endian uint (number of bytes depends on system).
    fn write_le_uint(&mut self, n: uint) -> IoResult<()> {
        extensions::u64_to_le_bytes(n as u64, uint::BYTES, |v| self.write(v))
    }

    /// Write a little-endian int (number of bytes depends on system).
    fn write_le_int(&mut self, n: int) -> IoResult<()> {
        extensions::u64_to_le_bytes(n as u64, int::BYTES, |v| self.write(v))
    }

    /// Write a big-endian uint (number of bytes depends on system).
    fn write_be_uint(&mut self, n: uint) -> IoResult<()> {
        extensions::u64_to_be_bytes(n as u64, uint::BYTES, |v| self.write(v))
    }

    /// Write a big-endian int (number of bytes depends on system).
    fn write_be_int(&mut self, n: int) -> IoResult<()> {
        extensions::u64_to_be_bytes(n as u64, int::BYTES, |v| self.write(v))
    }

    /// Write a big-endian u64 (8 bytes).
    fn write_be_u64(&mut self, n: u64) -> IoResult<()> {
        extensions::u64_to_be_bytes(n, 8u, |v| self.write(v))
    }

    /// Write a big-endian u32 (4 bytes).
    fn write_be_u32(&mut self, n: u32) -> IoResult<()> {
        extensions::u64_to_be_bytes(n as u64, 4u, |v| self.write(v))
    }

    /// Write a big-endian u16 (2 bytes).
    fn write_be_u16(&mut self, n: u16) -> IoResult<()> {
        extensions::u64_to_be_bytes(n as u64, 2u, |v| self.write(v))
    }

    /// Write a big-endian i64 (8 bytes).
    fn write_be_i64(&mut self, n: i64) -> IoResult<()> {
        extensions::u64_to_be_bytes(n as u64, 8u, |v| self.write(v))
    }

    /// Write a big-endian i32 (4 bytes).
    fn write_be_i32(&mut self, n: i32) -> IoResult<()> {
        extensions::u64_to_be_bytes(n as u64, 4u, |v| self.write(v))
    }

    /// Write a big-endian i16 (2 bytes).
    fn write_be_i16(&mut self, n: i16) -> IoResult<()> {
        extensions::u64_to_be_bytes(n as u64, 2u, |v| self.write(v))
    }

    /// Write a big-endian IEEE754 double-precision floating-point (8 bytes).
    fn write_be_f64(&mut self, f: f64) -> IoResult<()> {
        unsafe {
            self.write_be_u64(transmute(f))
        }
    }

    /// Write a big-endian IEEE754 single-precision floating-point (4 bytes).
    fn write_be_f32(&mut self, f: f32) -> IoResult<()> {
        unsafe {
            self.write_be_u32(transmute(f))
        }
    }

    /// Write a little-endian u64 (8 bytes).
    fn write_le_u64(&mut self, n: u64) -> IoResult<()> {
        extensions::u64_to_le_bytes(n, 8u, |v| self.write(v))
    }

    /// Write a little-endian u32 (4 bytes).
    fn write_le_u32(&mut self, n: u32) -> IoResult<()> {
        extensions::u64_to_le_bytes(n as u64, 4u, |v| self.write(v))
    }

    /// Write a little-endian u16 (2 bytes).
    fn write_le_u16(&mut self, n: u16) -> IoResult<()> {
        extensions::u64_to_le_bytes(n as u64, 2u, |v| self.write(v))
    }

    /// Write a little-endian i64 (8 bytes).
    fn write_le_i64(&mut self, n: i64) -> IoResult<()> {
        extensions::u64_to_le_bytes(n as u64, 8u, |v| self.write(v))
    }

    /// Write a little-endian i32 (4 bytes).
    fn write_le_i32(&mut self, n: i32) -> IoResult<()> {
        extensions::u64_to_le_bytes(n as u64, 4u, |v| self.write(v))
    }

    /// Write a little-endian i16 (2 bytes).
    fn write_le_i16(&mut self, n: i16) -> IoResult<()> {
        extensions::u64_to_le_bytes(n as u64, 2u, |v| self.write(v))
    }

    /// Write a little-endian IEEE754 double-precision floating-point
    /// (8 bytes).
    fn write_le_f64(&mut self, f: f64) -> IoResult<()> {
        unsafe {
            self.write_le_u64(transmute(f))
        }
    }

    /// Write a little-endian IEEE754 single-precision floating-point
    /// (4 bytes).
    fn write_le_f32(&mut self, f: f32) -> IoResult<()> {
        unsafe {
            self.write_le_u32(transmute(f))
        }
    }

    /// Write a u8 (1 byte).
    fn write_u8(&mut self, n: u8) -> IoResult<()> {
        self.write([n])
    }

    /// Write an i8 (1 byte).
    fn write_i8(&mut self, n: i8) -> IoResult<()> {
        self.write([n as u8])
    }

    /// Creates a wrapper around a mutable reference to the writer.
    ///
    /// This is useful to allow applying wrappers while still
    /// retaining ownership of the original value.
    fn by_ref<'a>(&'a mut self) -> RefWriter<'a, Self> {
        RefWriter { inner: self }
    }
}

impl Writer for Box<Writer> {
    fn write(&mut self, buf: &[u8]) -> IoResult<()> { self.write(buf) }
    fn flush(&mut self) -> IoResult<()> { self.flush() }
}

impl<'a> Writer for &'a mut Writer {
    fn write(&mut self, buf: &[u8]) -> IoResult<()> { self.write(buf) }
    fn flush(&mut self) -> IoResult<()> { self.flush() }
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
/// use std::io::{stdin, MemWriter};
///
/// let mut output = MemWriter::new();
///
/// {
///     // Don't give ownership of 'output' to the 'tee'. Instead we keep a
///     // handle to it in the outer scope
///     let mut tee = TeeReader::new(stdin(), output.by_ref());
///     process_input(tee);
/// }
///
/// println!("input processed: {}", output.unwrap());
/// # }
/// ```
pub struct RefWriter<'a, W> {
    /// The underlying writer which this is referencing
    inner: &'a mut W
}

impl<'a, W: Writer> Writer for RefWriter<'a, W> {
    fn write(&mut self, buf: &[u8]) -> IoResult<()> { self.inner.write(buf) }
    fn flush(&mut self) -> IoResult<()> { self.inner.flush() }
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
pub struct Lines<'r, T> {
    buffer: &'r mut T,
}

impl<'r, T: Buffer> Iterator<IoResult<~str>> for Lines<'r, T> {
    fn next(&mut self) -> Option<IoResult<~str>> {
        match self.buffer.read_line() {
            Ok(x) => Some(Ok(x)),
            Err(IoError { kind: EndOfFile, ..}) => None,
            Err(y) => Some(Err(y))
        }
    }
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
pub struct Chars<'r, T> {
    buffer: &'r mut T
}

impl<'r, T: Buffer> Iterator<IoResult<char>> for Chars<'r, T> {
    fn next(&mut self) -> Option<IoResult<char>> {
        match self.buffer.read_char() {
            Ok(x) => Some(Ok(x)),
            Err(IoError { kind: EndOfFile, ..}) => None,
            Err(y) => Some(Err(y))
        }
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
    /// encoded unicode codepoints. If a newline is encountered, then the
    /// newline is contained in the returned string.
    ///
    /// # Example
    ///
    /// ```rust
    /// use std::io;
    ///
    /// let mut reader = io::stdin();
    /// let input = reader.read_line().ok().unwrap_or("nothing".to_owned());
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
    fn read_line(&mut self) -> IoResult<~str> {
        self.read_until('\n' as u8).and_then(|line|
            match str::from_utf8(line.as_slice()) {
                Some(s) => Ok(s.to_owned()),
                None => Err(standard_error(InvalidInput)),
            }
        )
    }

    /// Create an iterator that reads a line on each iteration until EOF.
    ///
    /// # Error
    ///
    /// Any error other than `EndOfFile` that is produced by the underlying Reader
    /// is returned by the iterator and should be handled by the caller.
    fn lines<'r>(&'r mut self) -> Lines<'r, Self> {
        Lines { buffer: self }
    }

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
    fn read_until(&mut self, byte: u8) -> IoResult<Vec<u8>> {
        let mut res = Vec::new();

        let mut used;
        loop {
            {
                let available = match self.fill_buf() {
                    Ok(n) => n,
                    Err(ref e) if res.len() > 0 && e.kind == EndOfFile => {
                        used = 0;
                        break
                    }
                    Err(e) => return Err(e)
                };
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
            self.consume(used);
        }
        self.consume(used);
        Ok(res)
    }

    /// Reads the next utf8-encoded character from the underlying stream.
    ///
    /// # Error
    ///
    /// If an I/O error occurs, or EOF, then this function will return `Err`.
    /// This function will also return error if the stream does not contain a
    /// valid utf-8 encoded codepoint as the next few bytes in the stream.
    fn read_char(&mut self) -> IoResult<char> {
        let first_byte = try!(self.read_byte());
        let width = str::utf8_char_width(first_byte);
        if width == 1 { return Ok(first_byte as char) }
        if width == 0 { return Err(standard_error(InvalidInput)) } // not utf8
        let mut buf = [first_byte, 0, 0, 0];
        {
            let mut start = 1;
            while start < width {
                match try!(self.read(buf.mut_slice(start, width))) {
                    n if n == width - start => break,
                    n if n < width - start => { start += n; }
                    _ => return Err(standard_error(InvalidInput)),
                }
            }
        }
        match str::from_utf8(buf.slice_to(width)) {
            Some(s) => Ok(s.char_at(0)),
            None => Err(standard_error(InvalidInput))
        }
    }

    /// Create an iterator that reads a utf8-encoded character on each iteration
    /// until EOF.
    ///
    /// # Error
    ///
    /// Any error other than `EndOfFile` that is produced by the underlying Reader
    /// is returned by the iterator and should be handled by the caller.
    fn chars<'r>(&'r mut self) -> Chars<'r, Self> {
        Chars { buffer: self }
    }
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
    fn incoming<'r>(&'r mut self) -> IncomingConnections<'r, Self> {
        IncomingConnections { inc: self }
    }
}

/// An infinite iterator over incoming connection attempts.
/// Calling `next` will block the task until a connection is attempted.
///
/// Since connection attempts can continue forever, this iterator always returns
/// `Some`. The `Some` contains the `IoResult` representing whether the
/// connection attempt was successful.  A successful connection will be wrapped
/// in `Ok`. A failed connection is represented as an `Err`.
pub struct IncomingConnections<'a, A> {
    inc: &'a mut A,
}

impl<'a, T, A: Acceptor<T>> Iterator<IoResult<T>> for IncomingConnections<'a, A> {
    fn next(&mut self) -> Option<IoResult<T>> {
        Some(self.inc.accept())
    }
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
pub fn standard_error(kind: IoErrorKind) -> IoError {
    let desc = match kind {
        EndOfFile => "end of file",
        IoUnavailable => "I/O is unavailable",
        InvalidInput => "invalid input",
        OtherIoError => "unknown I/O error",
        FileNotFound => "file not found",
        PermissionDenied => "permission denied",
        ConnectionFailed => "connection failed",
        Closed => "stream is closed",
        ConnectionRefused => "connection refused",
        ConnectionReset => "connection reset",
        ConnectionAborted => "connection aborted",
        NotConnected => "not connected",
        BrokenPipe => "broken pipe",
        PathAlreadyExists => "file exists",
        PathDoesntExist => "no such file",
        MismatchedFileTypeForOperation => "mismatched file type",
        ResourceUnavailable => "resource unavailable",
        TimedOut => "operation timed out",
        ShortWrite(..) => "short write",
        NoProgress => "no progress",
    };
    IoError {
        kind: kind,
        desc: desc,
        detail: None,
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
#[deriving(Eq, Show, Hash)]
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
/// # fn main() {}
/// # fn foo() {
/// let info = match Path::new("foo.txt").stat() {
///     Ok(stat) => stat,
///     Err(e) => fail!("couldn't read foo.txt: {}", e),
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

bitflags!(
    #[doc="A set of permissions for a file or directory is represented
by a set of flags which are or'd together."]
    #[deriving(Hash)]
    #[deriving(Show)]
    flags FilePermission: u32 {
        static UserRead     = 0o400,
        static UserWrite    = 0o200,
        static UserExecute  = 0o100,
        static GroupRead    = 0o040,
        static GroupWrite   = 0o020,
        static GroupExecute = 0o010,
        static OtherRead    = 0o004,
        static OtherWrite   = 0o002,
        static OtherExecute = 0o001,

        static UserRWX  = UserRead.bits | UserWrite.bits | UserExecute.bits,
        static GroupRWX = GroupRead.bits | GroupWrite.bits | GroupExecute.bits,
        static OtherRWX = OtherRead.bits | OtherWrite.bits | OtherExecute.bits,

        #[doc="Permissions for user owned files, equivalent to 0644 on
unix-like systems."]
        static UserFile = UserRead.bits | UserWrite.bits | GroupRead.bits | OtherRead.bits,

        #[doc="Permissions for user owned directories, equivalent to 0755 on
unix-like systems."]
        static UserDir  = UserRWX.bits | GroupRead.bits | GroupExecute.bits |
                   OtherRead.bits | OtherExecute.bits,

        #[doc="Permissions for user owned executables, equivalent to 0755
on unix-like systems."]
        static UserExec = UserDir.bits,

        #[doc="All possible permissions enabled."]
        static AllPermissions = UserRWX.bits | GroupRWX.bits | OtherRWX.bits
    }
)

#[cfg(test)]
mod tests {
    use super::{IoResult, Reader, MemReader, NoProgress, InvalidInput};
    use prelude::*;
    use uint;

    #[deriving(Clone, Eq, Show)]
    enum BadReaderBehavior {
        GoodBehavior(uint),
        BadBehavior(uint)
    }

    struct BadReader<T> {
        r: T,
        behavior: Vec<BadReaderBehavior>,
    }

    impl<T: Reader> BadReader<T> {
        fn new(r: T, behavior: Vec<BadReaderBehavior>) -> BadReader<T> {
            BadReader { behavior: behavior, r: r }
        }
    }

    impl<T: Reader> Reader for BadReader<T> {
        fn read(&mut self, buf: &mut [u8]) -> IoResult<uint> {
            let BadReader { ref mut behavior, ref mut r } = *self;
            loop {
                if behavior.is_empty() {
                    // fall back on good
                    return r.read(buf);
                }
                match behavior.as_mut_slice()[0] {
                    GoodBehavior(0) => (),
                    GoodBehavior(ref mut x) => {
                        *x -= 1;
                        return r.read(buf);
                    }
                    BadBehavior(0) => (),
                    BadBehavior(ref mut x) => {
                        *x -= 1;
                        return Ok(0);
                    }
                };
                behavior.shift();
            }
        }
    }

    #[test]
    fn test_read_at_least() {
        let mut r = BadReader::new(MemReader::new(Vec::from_slice(bytes!("hello, world!"))),
                                   Vec::from_slice([GoodBehavior(uint::MAX)]));
        let mut buf = [0u8, ..5];
        assert!(r.read_at_least(1, buf).unwrap() >= 1);
        assert!(r.read_exact(5).unwrap().len() == 5); // read_exact uses read_at_least
        assert!(r.read_at_least(0, buf).is_ok());

        let mut r = BadReader::new(MemReader::new(Vec::from_slice(bytes!("hello, world!"))),
                                   Vec::from_slice([BadBehavior(50), GoodBehavior(uint::MAX)]));
        assert!(r.read_at_least(1, buf).unwrap() >= 1);

        let mut r = BadReader::new(MemReader::new(Vec::from_slice(bytes!("hello, world!"))),
                                   Vec::from_slice([BadBehavior(1), GoodBehavior(1),
                                                    BadBehavior(50), GoodBehavior(uint::MAX)]));
        assert!(r.read_at_least(1, buf).unwrap() >= 1);
        assert!(r.read_at_least(1, buf).unwrap() >= 1);

        let mut r = BadReader::new(MemReader::new(Vec::from_slice(bytes!("hello, world!"))),
                                   Vec::from_slice([BadBehavior(uint::MAX)]));
        assert_eq!(r.read_at_least(1, buf).unwrap_err().kind, NoProgress);

        let mut r = MemReader::new(Vec::from_slice(bytes!("hello, world!")));
        assert_eq!(r.read_at_least(5, buf).unwrap(), 5);
        assert_eq!(r.read_at_least(6, buf).unwrap_err().kind, InvalidInput);
    }

    #[test]
    fn test_push_at_least() {
        let mut r = BadReader::new(MemReader::new(Vec::from_slice(bytes!("hello, world!"))),
                                   Vec::from_slice([GoodBehavior(uint::MAX)]));
        let mut buf = Vec::new();
        assert!(r.push_at_least(1, 5, &mut buf).unwrap() >= 1);
        assert!(r.push_at_least(0, 5, &mut buf).is_ok());

        let mut r = BadReader::new(MemReader::new(Vec::from_slice(bytes!("hello, world!"))),
                                   Vec::from_slice([BadBehavior(50), GoodBehavior(uint::MAX)]));
        assert!(r.push_at_least(1, 5, &mut buf).unwrap() >= 1);

        let mut r = BadReader::new(MemReader::new(Vec::from_slice(bytes!("hello, world!"))),
                                   Vec::from_slice([BadBehavior(1), GoodBehavior(1),
                                                    BadBehavior(50), GoodBehavior(uint::MAX)]));
        assert!(r.push_at_least(1, 5, &mut buf).unwrap() >= 1);
        assert!(r.push_at_least(1, 5, &mut buf).unwrap() >= 1);

        let mut r = BadReader::new(MemReader::new(Vec::from_slice(bytes!("hello, world!"))),
                                   Vec::from_slice([BadBehavior(uint::MAX)]));
        assert_eq!(r.push_at_least(1, 5, &mut buf).unwrap_err().kind, NoProgress);

        let mut r = MemReader::new(Vec::from_slice(bytes!("hello, world!")));
        assert_eq!(r.push_at_least(5, 1, &mut buf).unwrap_err().kind, InvalidInput);
    }
}
