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
Types that implement both `Reader` and `Writer` and called 'streams',
and automatically implement trait `Stream`.
Implementations are provided for common I/O streams like
file, TCP, UDP, Unix domain sockets.
Readers and Writers may be composed to add capabilities like string
parsing, encoding, and compression.

This will likely live in std::io, not std::rt::io.

# Examples

Some examples of obvious things you might want to do

* Read lines from stdin

    for stdin().each_line |line| {
        println(line)
    }

* Read a complete file to a string, (converting newlines?)

    let contents = File::open("message.txt").read_to_str(); // read_to_str??

* Write a line to a file

    let file = File::open("message.txt", Create, Write);
    file.write_line("hello, file!");

* Iterate over the lines of a file

    do File::open("message.txt").each_line |line| {
        println(line)
    }

* Pull the lines of a file into a vector of strings

    let lines = File::open("message.txt").line_iter().to_vec();

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

For blocking (but possibly more efficient) implementations, look
in the `io::native` module.

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
* File vs. FileStream? File is shorter but could also be used for getting file info
  - maybe File is for general file querying and *also* has a static `open` method
* open vs. connect for generic stream opening
* Do we need `close` at all? dtors might be good enough
* How does I/O relate to the Iterator trait?
* std::base64 filters
* Using conditions is a big unknown since we don't have much experience with them
* Too many uses of OtherIoError

*/

use prelude::*;
use to_str::ToStr;
use str::{StrSlice, OwnedStr};

// Reexports
pub use self::stdio::stdin;
pub use self::stdio::stdout;
pub use self::stdio::stderr;
pub use self::stdio::print;
pub use self::stdio::println;

pub use self::file::FileStream;
pub use self::timer::Timer;
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

/// Synchronous, in-memory I/O.
pub mod pipe;

/// Synchronous, non-blocking network I/O.
pub mod net {
    pub mod tcp;
    pub mod udp;
    pub mod ip;
    #[cfg(unix)]
    pub mod unix;
}

/// Readers and Writers for memory buffers and strings.
pub mod mem;

/// Non-blocking access to stdin, stdout, stderr
pub mod stdio;

/// Implementations for Option
mod option;

/// Basic stream compression. XXX: Belongs with other flate code
pub mod flate;

/// Interop between byte streams and pipes. Not sure where it belongs
pub mod comm_adapters;

/// Extension traits
mod extensions;

/// Non-I/O things needed by the I/O module
mod support;

/// Basic Timer
pub mod timer;

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

/// Mock implementations for testing
mod mock;

/// The default buffer size for various I/O operations
/// XXX: Not pub
pub static DEFAULT_BUF_SIZE: uint = 1024 * 64;

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
    BrokenPipe
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
            BrokenPipe => ~"BrokenPipe"
        }
    }
}

// XXX: Can't put doc comments on macros
// Raised by `I/O` operations on error.
condition! {
    // FIXME (#6009): uncomment `pub` after expansion support lands.
    /*pub*/ io_error: super::IoError -> ();
}

// XXX: Can't put doc comments on macros
// Raised by `read` on error
condition! {
    // FIXME (#6009): uncomment `pub` after expansion support lands.
    /*pub*/ read_error: super::IoError -> ();
}

pub trait Reader {
    /// Read bytes, up to the length of `buf` and place them in `buf`.
    /// Returns the number of bytes read. The number of bytes read my
    /// be less than the number requested, even 0. Returns `None` on EOF.
    ///
    /// # Failure
    ///
    /// Raises the `read_error` condition on error. If the condition
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
    ///     let reader = FileStream::new()
    ///     while !reader.eof() {
    ///         println(reader.read_line());
    ///     }
    ///
    /// # Failure
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

pub trait Stream: Reader + Writer { }

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
        EndOfFile => {
            IoError {
                kind: EndOfFile,
                desc: "End of file",
                detail: None
            }
        }
        _ => fail!()
    }
}

pub fn placeholder_error() -> IoError {
    IoError {
        kind: OtherIoError,
        desc: "Placeholder error. You shouldn't be seeing this",
        detail: None
    }
}

/// Instructions on how to open a file and return a `FileStream`.
pub enum FileMode {
    /// Opens an existing file. IoError if file does not exist.
    Open,
    /// Creates a file. IoError if file exists.
    Create,
    /// Opens an existing file or creates a new one.
    OpenOrCreate,
    /// Opens an existing file or creates a new one, positioned at EOF.
    Append,
    /// Opens an existing file, truncating it to 0 bytes.
    Truncate,
    /// Opens an existing file or creates a new one, truncating it to 0 bytes.
    CreateOrTruncate,
}

/// Access permissions with which the file should be opened.
/// `FileStream`s opened with `Read` will raise an `io_error` condition if written to.
pub enum FileAccess {
    Read,
    Write,
    ReadWrite
}
