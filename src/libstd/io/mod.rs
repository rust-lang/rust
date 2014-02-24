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
        print!("{}", line);
    }
    ```

* Read a complete file

    ```rust
    use std::io::File;

    let contents = File::open(&Path::new("message.txt")).read_to_end();
    ```

* Write a line to a file

    ```rust
    # #[allow(unused_must_use)];
    use std::io::File;

    let mut file = File::create(&Path::new("message.txt"));
    file.write(bytes!("hello, file!\n"));
    # drop(file);
    # ::std::io::fs::unlink(&Path::new("message.txt"));
    ```

* Iterate over the lines of a file

    ```rust
    use std::io::BufferedReader;
    use std::io::File;

    let path = Path::new("message.txt");
    let mut file = BufferedReader::new(File::open(&path));
    for line in file.lines() {
        print!("{}", line);
    }
    ```

* Pull the lines of a file into a vector of strings

    ```rust
    use std::io::BufferedReader;
    use std::io::File;

    let path = Path::new("message.txt");
    let mut file = BufferedReader::new(File::open(&path));
    let lines: ~[~str] = file.lines().collect();
    ```

* Make a simple TCP client connection and request

    ```rust,should_fail
    # #[allow(unused_must_use)];
    use std::io::net::ip::SocketAddr;
    use std::io::net::tcp::TcpStream;

    let addr = from_str::<SocketAddr>("127.0.0.1:8080").unwrap();
    let mut socket = TcpStream::connect(addr).unwrap();
    socket.write(bytes!("GET / HTTP/1.0\n\n"));
    let response = socket.read_to_end();
    ```

* Make a simple TCP server

    ```rust
    # fn main() { }
    # fn foo() {
    # #[allow(unused_must_use, dead_code)];
    use std::io::net::tcp::TcpListener;
    use std::io::net::ip::{Ipv4Addr, SocketAddr};
    use std::io::{Acceptor, Listener};

    let addr = SocketAddr { ip: Ipv4Addr(127, 0, 0, 1), port: 80 };
    let listener = TcpListener::bind(addr);

    // bind the listener to the specified address
    let mut acceptor = listener.listen();

    // accept connections and process them
    # fn handle_client<T>(_: T) {}
    for stream in acceptor.incoming() {
        spawn(proc() {
            handle_client(stream);
        });
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
# #[allow(unused_must_use)];
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

*/

#[allow(missing_doc)];
#[deny(unused_must_use)];

use cast;
use char::Char;
use container::Container;
use fmt;
use int;
use iter::Iterator;
use option::{Option, Some, None};
use path::Path;
use result::{Ok, Err, Result};
use str::{StrSlice, OwnedStr};
use str;
use to_str::ToStr;
use uint;
use unstable::finally::try_finally;
use vec::{OwnedVector, MutableVector, ImmutableVector, OwnedCloneableVector};
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
pub use self::process::{Process, ProcessConfig};

pub use self::mem::{MemReader, BufReader, MemWriter, BufWriter};
pub use self::buffered::{BufferedReader, BufferedWriter, BufferedStream,
                         LineBufferedWriter};
pub use self::comm_adapters::{PortReader, ChanWriter};

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
mod mem;

/// Non-blocking access to stdin, stdout, stderr
pub mod stdio;

/// Implementations for Result
mod result;

/// Extension traits
pub mod extensions;

/// Basic Timer
pub mod timer;

/// Buffered I/O wrappers
mod buffered;

/// Signal handling
pub mod signal;

/// Utility implementations of Reader and Writer
pub mod util;

/// Adapatation of Chan/Port types to a Writer/Reader type.
mod comm_adapters;

/// The default buffer size for various I/O operations
// libuv recommends 64k buffers to maximize throughput
// https://groups.google.com/forum/#!topic/libuv/oQO1HJAIDdA
static DEFAULT_BUF_SIZE: uint = 1024 * 64;

pub type IoResult<T> = Result<T, IoError>;

/// The type passed to I/O condition handlers to indicate error
///
/// # FIXME
///
/// Is something like this sufficient? It's kind of archaic
#[deriving(Eq, Clone)]
pub struct IoError {
    kind: IoErrorKind,
    desc: &'static str,
    detail: Option<~str>
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

#[deriving(Eq, Clone)]
pub enum IoErrorKind {
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

pub trait Reader {

    // Only method which need to get implemented for this trait

    /// Read bytes, up to the length of `buf` and place them in `buf`.
    /// Returns the number of bytes read. The number of bytes read my
    /// be less than the number requested, even 0. Returns `Err` on EOF.
    ///
    /// # Error
    ///
    /// If an error occurs during this I/O operation, then it is returned as
    /// `Err(IoError)`. Note that end-of-file is considered an error, and can be
    /// inspected for in the error's `kind` field. Also note that reading 0
    /// bytes is not considered an error in all circumstances
    fn read(&mut self, buf: &mut [u8]) -> IoResult<uint>;

    // Convenient helper methods based on the above methods

    /// Reads a single byte. Returns `Err` on EOF.
    fn read_byte(&mut self) -> IoResult<u8> {
        let mut buf = [0];
        loop {
            match self.read(buf) {
                Ok(0) => {
                    debug!("read 0 bytes. trying again");
                }
                Ok(1) => return Ok(buf[0]),
                Ok(_) => unreachable!(),
                Err(e) => return Err(e)
            }
        }
    }

    /// Reads `len` bytes and appends them to a vector.
    ///
    /// May push fewer than the requested number of bytes on error
    /// or EOF. If `Ok(())` is returned, then all of the requested bytes were
    /// pushed on to the vector, otherwise the amount `len` bytes couldn't be
    /// read (an error was encountered), and the error is returned.
    fn push_bytes(&mut self, buf: &mut ~[u8], len: uint) -> IoResult<()> {
        struct State<'a> {
            buf: &'a mut ~[u8],
            total_read: uint
        }

        let start_len = buf.len();
        let mut s = State { buf: buf, total_read: 0 };

        s.buf.reserve_additional(len);
        unsafe { s.buf.set_len(start_len + len); }

        try_finally(
            &mut s, (),
            |s, _| {
                while s.total_read < len {
                    let len = s.buf.len();
                    let slice = s.buf.mut_slice(start_len + s.total_read, len);
                    match self.read(slice) {
                        Ok(nread) => {
                            s.total_read += nread;
                        }
                        Err(e) => return Err(e)
                    }
                }
                Ok(())
            },
            |s| unsafe { s.buf.set_len(start_len + s.total_read) })
    }

    /// Reads `len` bytes and gives you back a new vector of length `len`
    ///
    /// # Error
    ///
    /// Fails with the same conditions as `read`. Additionally returns error on
    /// on EOF. Note that if an error is returned, then some number of bytes may
    /// have already been consumed from the underlying reader, and they are lost
    /// (not returned as part of the error). If this is unacceptable, then it is
    /// recommended to use the `push_bytes` or `read` methods.
    fn read_bytes(&mut self, len: uint) -> IoResult<~[u8]> {
        let mut buf = vec::with_capacity(len);
        match self.push_bytes(&mut buf, len) {
            Ok(()) => Ok(buf),
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
    fn read_to_end(&mut self) -> IoResult<~[u8]> {
        let mut buf = vec::with_capacity(DEFAULT_BUF_SIZE);
        loop {
            match self.push_bytes(&mut buf, DEFAULT_BUF_SIZE) {
                Ok(()) => {}
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
            match str::from_utf8_owned(s) {
                Some(s) => Ok(s),
                None => Err(standard_error(InvalidInput)),
            }
        })
    }

    /// Create an iterator that reads a single byte on
    /// each iteration, until EOF.
    ///
    /// # Error
    ///
    /// The iterator protocol causes all specifics about errors encountered to
    /// be swallowed. All errors will be signified by returning `None` from the
    /// iterator. If this is undesirable, it is recommended to use the
    /// `read_byte` method.
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
    /// The number of bytes returned is system-dependant.
    fn read_le_uint(&mut self) -> IoResult<uint> {
        self.read_le_uint_n(uint::BYTES).map(|i| i as uint)
    }

    /// Reads a little-endian integer.
    ///
    /// The number of bytes returned is system-dependant.
    fn read_le_int(&mut self) -> IoResult<int> {
        self.read_le_int_n(int::BYTES).map(|i| i as int)
    }

    /// Reads a big-endian unsigned integer.
    ///
    /// The number of bytes returned is system-dependant.
    fn read_be_uint(&mut self) -> IoResult<uint> {
        self.read_be_uint_n(uint::BYTES).map(|i| i as uint)
    }

    /// Reads a big-endian integer.
    ///
    /// The number of bytes returned is system-dependant.
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
            cast::transmute::<u64, f64>(i)
        })
    }

    /// Reads a big-endian `f32`.
    ///
    /// `f32`s are 4 byte, IEEE754 single-precision floating point numbers.
    fn read_be_f32(&mut self) -> IoResult<f32> {
        self.read_be_u32().map(|i| unsafe {
            cast::transmute::<u32, f32>(i)
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
            cast::transmute::<u64, f64>(i)
        })
    }

    /// Reads a little-endian `f32`.
    ///
    /// `f32`s are 4 byte, IEEE754 single-precision floating point numbers.
    fn read_le_f32(&mut self) -> IoResult<f32> {
        self.read_le_u32().map(|i| unsafe {
            cast::transmute::<u32, f32>(i)
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

impl Reader for ~Reader {
    fn read(&mut self, buf: &mut [u8]) -> IoResult<uint> { self.read(buf) }
}

impl<'a> Reader for &'a mut Reader {
    fn read(&mut self, buf: &mut [u8]) -> IoResult<uint> { self.read(buf) }
}

pub struct RefReader<'a, R> {
    priv inner: &'a mut R
}

impl<'a, R: Reader> Reader for RefReader<'a, R> {
    fn read(&mut self, buf: &mut [u8]) -> IoResult<uint> { self.inner.read(buf) }
}

fn extend_sign(val: u64, nbytes: uint) -> i64 {
    let shift = (8 - nbytes) * 8;
    (val << shift) as i64 >> shift
}

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
            self.write_be_u64(cast::transmute(f))
        }
    }

    /// Write a big-endian IEEE754 single-precision floating-point (4 bytes).
    fn write_be_f32(&mut self, f: f32) -> IoResult<()> {
        unsafe {
            self.write_be_u32(cast::transmute(f))
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
            self.write_le_u64(cast::transmute(f))
        }
    }

    /// Write a little-endian IEEE754 single-precision floating-point
    /// (4 bytes).
    fn write_le_f32(&mut self, f: f32) -> IoResult<()> {
        unsafe {
            self.write_le_u32(cast::transmute(f))
        }
    }

    /// Write a u8 (1 byte).
    fn write_u8(&mut self, n: u8) -> IoResult<()> {
        self.write([n])
    }

    /// Write a i8 (1 byte).
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

impl Writer for ~Writer {
    fn write(&mut self, buf: &[u8]) -> IoResult<()> { self.write(buf) }
    fn flush(&mut self) -> IoResult<()> { self.flush() }
}

impl<'a> Writer for &'a mut Writer {
    fn write(&mut self, buf: &[u8]) -> IoResult<()> { self.write(buf) }
    fn flush(&mut self) -> IoResult<()> { self.flush() }
}

pub struct RefWriter<'a, W> {
    inner: &'a mut W
}

impl<'a, W: Writer> Writer for RefWriter<'a, W> {
    fn write(&mut self, buf: &[u8]) -> IoResult<()> { self.inner.write(buf) }
    fn flush(&mut self) -> IoResult<()> { self.inner.flush() }
}


pub trait Stream: Reader + Writer { }

impl<T: Reader + Writer> Stream for T {}

/// An iterator that reads a line on each iteration,
/// until `.read_line()` returns `None`.
///
/// # Notes about the Iteration Protocol
///
/// The `Lines` may yield `None` and thus terminate
/// an iteration, but continue to yield elements if iteration
/// is attempted again.
///
/// # Error
///
/// This iterator will swallow all I/O errors, transforming `Err` values to
/// `None`. If errors need to be handled, it is recommended to use the
/// `read_line` method directly.
pub struct Lines<'r, T> {
    priv buffer: &'r mut T,
}

impl<'r, T: Buffer> Iterator<~str> for Lines<'r, T> {
    fn next(&mut self) -> Option<~str> {
        self.buffer.read_line().ok()
    }
}

/// An iterator that reads a utf8-encoded character on each iteration,
/// until `.read_char()` returns `None`.
///
/// # Notes about the Iteration Protocol
///
/// The `Chars` may yield `None` and thus terminate
/// an iteration, but continue to yield elements if iteration
/// is attempted again.
///
/// # Error
///
/// This iterator will swallow all I/O errors, transforming `Err` values to
/// `None`. If errors need to be handled, it is recommended to use the
/// `read_char` method directly.
pub struct Chars<'r, T> {
    priv buffer: &'r mut T
}

impl<'r, T: Buffer> Iterator<char> for Chars<'r, T> {
    fn next(&mut self) -> Option<char> {
        self.buffer.read_char().ok()
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
    fn fill<'a>(&'a mut self) -> IoResult<&'a [u8]>;

    /// Tells this buffer that `amt` bytes have been consumed from the buffer,
    /// so they should no longer be returned in calls to `fill` or `read`.
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
    /// let input = reader.read_line().ok().unwrap_or(~"nothing");
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
            match str::from_utf8_owned(line) {
                Some(s) => Ok(s),
                None => Err(standard_error(InvalidInput)),
            }
        )
    }

    /// Create an iterator that reads a line on each iteration until EOF.
    ///
    /// # Error
    ///
    /// This iterator will transform all error values to `None`, discarding the
    /// cause of the error. If this is undesirable, it is recommended to call
    /// `read_line` directly.
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
    fn read_until(&mut self, byte: u8) -> IoResult<~[u8]> {
        let mut res = ~[];

        let mut used;
        loop {
            {
                let available = match self.fill() {
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

    /// Create an iterator that reads a utf8-encoded character on each iteration until EOF.
    ///
    /// # Error
    ///
    /// This iterator will transform all error values to `None`, discarding the
    /// cause of the error. If this is undesirable, it is recommended to call
    /// `read_char` directly.
    fn chars<'r>(&'r mut self) -> Chars<'r, Self> {
        Chars { buffer: self }
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
    priv inc: &'a mut A,
}

impl<'a, T, A: Acceptor<T>> Iterator<IoResult<T>> for IncomingConnections<'a, A> {
    fn next(&mut self) -> Option<IoResult<T>> {
        Some(self.inc.accept())
    }
}

pub fn standard_error(kind: IoErrorKind) -> IoError {
    let desc = match kind {
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
/// opened with `Read` will return an error if written to.
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
