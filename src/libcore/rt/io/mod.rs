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

This module defines the Rust interface for synchronous I/O.  It is
build around Reader and Writer traits that define byte stream sources
and sinks.  Implementations are provided for common I/O streams like
file, TCP, UDP, Unix domain sockets, multiple types of memory bufers.
Readers and Writers may be composed to add things like string parsing,
and compression.

This will likely live in core::io, not core::rt::io.

# Examples

Some examples of obvious things you might want to do

* Read lines from stdin

    for stdin().each_line |line| {
        println(line)
    }

* Read a complete file to a string, (converting newlines?)

    let contents = FileStream::open("message.txt").read_to_str(); // read_to_str??

* Write a line to a file

    let file = FileStream::open("message.txt", Create, Write);
    file.write_line("hello, file!");

* Iterate over the lines of a file

    do FileStream::open("message.txt").each_line |line| {
        println(line)
    }

* Pull the lines of a file into a vector of strings

    let lines = FileStream::open("message.txt").line_iter().to_vec();

* Make an simple HTTP request

    let socket = TcpStream::open("localhost:8080");
    socket.write_line("GET / HTTP/1.0");
    socket.write_line("");
    let response = socket.read_to_end();

* Connect based on URL? Requires thinking about where the URL type lives
  and how to make protocol handlers extensible, e.g. the "tcp" protocol
  yields a `TcpStream`.

    connect("tcp://localhost:8080");

# Terms

* reader
* writer
* stream
* Blocking vs. non-blocking
* synchrony and asynchrony

I tend to call this implementation non-blocking, because performing I/O
doesn't block the progress of other tasks. Is that how we want to present
it, 'synchronous but non-blocking'?

# Error Handling

# Resource management

* `close` vs. RAII

# Paths and URLs

# std

Some I/O things don't belong in core

  - url
  - net - `fn connect`
    - http
  - flate

# XXX

* Should default constructors take `Path` or `&str`? `Path` makes simple cases verbose.
  Overloading would be nice.
* Add overloading for Path and &str and Url &str
* stdin/err/out
* print, println, etc.
* fsync
* relationship with filesystem querying, Directory, File types etc.
* Rename Reader/Writer to ByteReader/Writer, make Reader/Writer generic?
* Trait for things that are both readers and writers, Stream?
* How to handle newline conversion
* String conversion
* File vs. FileStream? File is shorter but could also be used for getting file info
  - maybe File is for general file querying and *also* has a static `open` method
* open vs. connect for generic stream opening
* Do we need `close` at all? dtors might be good enough
* How does I/O relate to the Iterator trait?
* std::base64 filters

*/

use prelude::*;

// Reexports
pub use self::stdio::stdin;
pub use self::stdio::stdout;
pub use self::stdio::stderr;
pub use self::stdio::print;
pub use self::stdio::println;

pub use self::file::FileStream;
pub use self::net::ip::IpAddr;
pub use self::net::tcp::TcpListener;
pub use self::net::tcp::TcpStream;
pub use self::net::udp::UdpStream;

// Some extension traits that all Readers and Writers get.
pub use self::extensions::ReaderUtil;
pub use self::extensions::ReaderByteConversions;
pub use self::extensions::WriterByteConversions;

/// Synchronous, non-blocking file I/O.
pub mod file;

/// Synchronous, non-blocking network I/O.
pub mod net {
    pub mod tcp;
    pub mod udp;
    pub mod ip;
    #[cfg(unix)]
    pub mod unix;
    pub mod http;
}

/// Readers and Writers for memory buffers and strings.
#[cfg(not(stage0))] // XXX Using unsnapshotted features
pub mod mem;

/// Non-blocking access to stdin, stdout, stderr
pub mod stdio;

/// Implementations for Option
mod option;

/// Basic stream compression. XXX: Belongs with other flate code
#[cfg(not(stage0))] // XXX Using unsnapshotted features
pub mod flate;

/// Interop between byte streams and pipes. Not sure where it belongs
#[cfg(not(stage0))] // XXX "
pub mod comm_adapters;

/// Extension traits
mod extensions;

/// Non-I/O things needed by the I/O module
mod support;

/// Thread-blocking implementations
pub mod native {
    /// Posix file I/O
    pub mod file;
    /// # XXX - implement this
    pub mod stdio { }
    /// Sockets
    /// # XXX - implement this
    pub mod net {
        pub mod tcp { }
        pub mod udp { }
        #[cfg(unix)]
        pub mod unix { }
    }
}


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

#[deriving(Eq)]
pub enum IoErrorKind {
    FileNotFound,
    FilePermission,
    ConnectionFailed,
    Closed,
    OtherIoError,
    PreviousIoError
}

// XXX: Can't put doc comments on macros
// Raised by `I/O` operations on error.
condition! {
    io_error: super::IoError -> ();
}

pub trait Reader {
    /// Read bytes, up to the length of `buf` and place them in `buf`.
    /// Returns the number of bytes read, or `None` on EOF.
    ///
    /// # Failure
    ///
    /// Raises the `io_error` condition on error, then returns `None`.
    ///
    /// # XXX
    ///
    /// This doesn't take a `len` argument like the old `read`.
    /// Will people often need to slice their vectors to call this
    /// and will that be annoying?
    fn read(&mut self, buf: &mut [u8]) -> Option<uint>;

    /// Return whether the Reader has reached the end of the stream.
    ///
    /// # Example
    ///
    ///     let reader = FileStream::new()
    ///     while !reader.eof() {
    ///         println(reader.read_line());
    ///     }
    ///
    /// # Failue
    ///
    /// Returns `true` on failure.
    fn eof(&mut self) -> bool;
}

pub trait Writer {
    /// Write the given buffer
    ///
    /// # Failure
    ///
    /// Raises the `io_error` condition on error
    fn write(&mut self, buf: &[u8]);

    /// Flush output
    fn flush(&mut self);
}

/// I/O types that may be closed
///
/// Any further operations performed on a closed resource will raise
/// on `io_error`
pub trait Close {
    /// Close the I/O resource
    fn close(&mut self);
}

pub trait Stream: Reader + Writer + Close { }

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

/// A listener is a value that listens for connections
pub trait Listener<S> {
    /// Wait for and accept an incoming connection
    ///
    /// Returns `None` on timeout.
    ///
    /// # Failure
    ///
    /// Raises `io_error` condition. If the condition is handled,
    /// then `accept` returns `None`.
    fn accept(&mut self) -> Option<S>;
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
    match kind {
        PreviousIoError => {
            IoError {
                kind: PreviousIoError,
                desc: "Failing due to a previous I/O error",
                detail: None
            }
        }
        _ => fail!()
    }
}
