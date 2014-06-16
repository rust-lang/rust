// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

/*! Non-blocking access to stdin, stdout, and stderr.

This module provides bindings to the local event loop's TTY interface, using it
to offer synchronous but non-blocking versions of stdio. These handles can be
inspected for information about terminal dimensions or for related information
about the stream or terminal to which it is attached.

# Example

```rust
# #![allow(unused_must_use)]
use std::io;

let mut out = io::stdout();
out.write(bytes!("Hello, world!"));
```

*/

use failure::local_stderr;
use fmt;
use io::{Reader, Writer, IoResult, IoError, OtherIoError,
         standard_error, EndOfFile, LineBufferedWriter, BufferedReader};
use iter::Iterator;
use kinds::Send;
use libc;
use option::{Option, Some, None};
use owned::Box;
use result::{Ok, Err};
use rt;
use rt::local::Local;
use rt::task::Task;
use rt::rtio::{DontClose, IoFactory, LocalIo, RtioFileStream, RtioTTY};
use slice::ImmutableVector;
use str::StrSlice;
use uint;

// And so begins the tale of acquiring a uv handle to a stdio stream on all
// platforms in all situations. Our story begins by splitting the world into two
// categories, windows and unix. Then one day the creators of unix said let
// there be redirection! And henceforth there was redirection away from the
// console for standard I/O streams.
//
// After this day, the world split into four factions:
//
// 1. Unix with stdout on a terminal.
// 2. Unix with stdout redirected.
// 3. Windows with stdout on a terminal.
// 4. Windows with stdout redirected.
//
// Many years passed, and then one day the nation of libuv decided to unify this
// world. After months of toiling, uv created three ideas: TTY, Pipe, File.
// These three ideas propagated throughout the lands and the four great factions
// decided to settle among them.
//
// The groups of 1, 2, and 3 all worked very hard towards the idea of TTY. Upon
// doing so, they even enhanced themselves further then their Pipe/File
// brethren, becoming the dominant powers.
//
// The group of 4, however, decided to work independently. They abandoned the
// common TTY belief throughout, and even abandoned the fledgling Pipe belief.
// The members of the 4th faction decided to only align themselves with File.
//
// tl;dr; TTY works on everything but when windows stdout is redirected, in that
//        case pipe also doesn't work, but magically file does!
enum StdSource {
    TTY(Box<RtioTTY + Send>),
    File(Box<RtioFileStream + Send>),
}

fn src<T>(fd: libc::c_int, readable: bool, f: |StdSource| -> T) -> T {
    LocalIo::maybe_raise(|io| {
        Ok(match io.tty_open(fd, readable) {
            Ok(tty) => f(TTY(tty)),
            Err(_) => f(File(io.fs_from_raw_fd(fd, DontClose))),
        })
    }).map_err(IoError::from_rtio_error).unwrap()
}

local_data_key!(local_stdout: Box<Writer + Send>)

/// Creates a new non-blocking handle to the stdin of the current process.
///
/// The returned handled is buffered by default with a `BufferedReader`. If
/// buffered access is not desired, the `stdin_raw` function is provided to
/// provided unbuffered access to stdin.
///
/// Care should be taken when creating multiple handles to the stdin of a
/// process. Because this is a buffered reader by default, it's possible for
/// pending input to be unconsumed in one reader and unavailable to other
/// readers. It is recommended that only one handle at a time is created for the
/// stdin of a process.
///
/// See `stdout()` for more notes about this function.
pub fn stdin() -> BufferedReader<StdReader> {
    // The default buffer capacity is 64k, but apparently windows doesn't like
    // 64k reads on stdin. See #13304 for details, but the idea is that on
    // windows we use a slightly smaller buffer that's been seen to be
    // acceptable.
    if cfg!(windows) {
        BufferedReader::with_capacity(8 * 1024, stdin_raw())
    } else {
        BufferedReader::new(stdin_raw())
    }
}

/// Creates a new non-blocking handle to the stdin of the current process.
///
/// Unlike `stdin()`, the returned reader is *not* a buffered reader.
///
/// See `stdout()` for more notes about this function.
pub fn stdin_raw() -> StdReader {
    src(libc::STDIN_FILENO, true, |src| StdReader { inner: src })
}

/// Creates a line-buffered handle to the stdout of the current process.
///
/// Note that this is a fairly expensive operation in that at least one memory
/// allocation is performed. Additionally, this must be called from a runtime
/// task context because the stream returned will be a non-blocking object using
/// the local scheduler to perform the I/O.
///
/// Care should be taken when creating multiple handles to an output stream for
/// a single process. While usage is still safe, the output may be surprising if
/// no synchronization is performed to ensure a sane output.
pub fn stdout() -> LineBufferedWriter<StdWriter> {
    LineBufferedWriter::new(stdout_raw())
}

/// Creates an unbuffered handle to the stdout of the current process
///
/// See notes in `stdout()` for more information.
pub fn stdout_raw() -> StdWriter {
    src(libc::STDOUT_FILENO, false, |src| StdWriter { inner: src })
}

/// Creates a line-buffered handle to the stderr of the current process.
///
/// See `stdout()` for notes about this function.
pub fn stderr() -> LineBufferedWriter<StdWriter> {
    LineBufferedWriter::new(stderr_raw())
}

/// Creates an unbuffered handle to the stderr of the current process
///
/// See notes in `stdout()` for more information.
pub fn stderr_raw() -> StdWriter {
    src(libc::STDERR_FILENO, false, |src| StdWriter { inner: src })
}

/// Resets the task-local stdout handle to the specified writer
///
/// This will replace the current task's stdout handle, returning the old
/// handle. All future calls to `print` and friends will emit their output to
/// this specified handle.
///
/// Note that this does not need to be called for all new tasks; the default
/// output handle is to the process's stdout stream.
pub fn set_stdout(stdout: Box<Writer + Send>) -> Option<Box<Writer + Send>> {
    local_stdout.replace(Some(stdout)).and_then(|mut s| {
        let _ = s.flush();
        Some(s)
    })
}

/// Resets the task-local stderr handle to the specified writer
///
/// This will replace the current task's stderr handle, returning the old
/// handle. Currently, the stderr handle is used for printing failure messages
/// during task failure.
///
/// Note that this does not need to be called for all new tasks; the default
/// output handle is to the process's stderr stream.
pub fn set_stderr(stderr: Box<Writer + Send>) -> Option<Box<Writer + Send>> {
    local_stderr.replace(Some(stderr)).and_then(|mut s| {
        let _ = s.flush();
        Some(s)
    })
}

// Helper to access the local task's stdout handle
//
// Note that this is not a safe function to expose because you can create an
// aliased pointer very easily:
//
//  with_task_stdout(|io1| {
//      with_task_stdout(|io2| {
//          // io1 aliases io2
//      })
//  })
fn with_task_stdout(f: |&mut Writer| -> IoResult<()>) {
    let result = if Local::exists(None::<Task>) {
        let mut my_stdout = local_stdout.replace(None).unwrap_or_else(|| {
            box stdout() as Box<Writer + Send>
        });
        let result = f(my_stdout);
        local_stdout.replace(Some(my_stdout));
        result
    } else {
        let mut io = rt::Stdout;
        f(&mut io as &mut Writer)
    };
    match result {
        Ok(()) => {}
        Err(e) => fail!("failed printing to stdout: {}", e),
    }
}

/// Flushes the local task's stdout handle.
///
/// By default, this stream is a line-buffering stream, so flushing may be
/// necessary to ensure that all output is printed to the screen (if there are
/// no newlines printed).
///
/// Note that logging macros do not use this stream. Using the logging macros
/// will emit output to stderr, and while they are line buffered the log
/// messages are always terminated in a newline (no need to flush).
pub fn flush() {
    with_task_stdout(|io| io.flush())
}

/// Prints a string to the stdout of the current process. No newline is emitted
/// after the string is printed.
pub fn print(s: &str) {
    with_task_stdout(|io| io.write(s.as_bytes()))
}

/// Prints a string as a line. to the stdout of the current process. A literal
/// `\n` character is printed to the console after the string.
pub fn println(s: &str) {
    with_task_stdout(|io| {
        io.write(s.as_bytes()).and_then(|()| io.write(['\n' as u8]))
    })
}

/// Similar to `print`, but takes a `fmt::Arguments` structure to be compatible
/// with the `format_args!` macro.
pub fn print_args(fmt: &fmt::Arguments) {
    with_task_stdout(|io| write!(io, "{}", fmt))
}

/// Similar to `println`, but takes a `fmt::Arguments` structure to be
/// compatible with the `format_args!` macro.
pub fn println_args(fmt: &fmt::Arguments) {
    with_task_stdout(|io| writeln!(io, "{}", fmt))
}

/// Representation of a reader of a standard input stream
pub struct StdReader {
    inner: StdSource
}

impl StdReader {
    /// Returns whether this stream is attached to a TTY instance or not.
    pub fn isatty(&self) -> bool {
        match self.inner {
            TTY(..) => true,
            File(..) => false,
        }
    }
}

impl Reader for StdReader {
    fn read(&mut self, buf: &mut [u8]) -> IoResult<uint> {
        let ret = match self.inner {
            TTY(ref mut tty) => {
                // Flush the task-local stdout so that weird issues like a
                // print!'d prompt not being shown until after the user hits
                // enter.
                flush();
                tty.read(buf)
            },
            File(ref mut file) => file.read(buf).map(|i| i as uint),
        }.map_err(IoError::from_rtio_error);
        match ret {
            // When reading a piped stdin, libuv will return 0-length reads when
            // stdin reaches EOF. For pretty much all other streams it will
            // return an actual EOF error, but apparently for stdin it's a
            // little different. Hence, here we convert a 0 length read to an
            // end-of-file indicator so the caller knows to stop reading.
            Ok(0) => { Err(standard_error(EndOfFile)) }
            ret @ Ok(..) | ret @ Err(..) => ret,
        }
    }
}

/// Representation of a writer to a standard output stream
pub struct StdWriter {
    inner: StdSource
}

impl StdWriter {
    /// Gets the size of this output window, if possible. This is typically used
    /// when the writer is attached to something like a terminal, this is used
    /// to fetch the dimensions of the terminal.
    ///
    /// If successful, returns `Ok((width, height))`.
    ///
    /// # Error
    ///
    /// This function will return an error if the output stream is not actually
    /// connected to a TTY instance, or if querying the TTY instance fails.
    pub fn winsize(&mut self) -> IoResult<(int, int)> {
        match self.inner {
            TTY(ref mut tty) => {
                tty.get_winsize().map_err(IoError::from_rtio_error)
            }
            File(..) => {
                Err(IoError {
                    kind: OtherIoError,
                    desc: "stream is not a tty",
                    detail: None,
                })
            }
        }
    }

    /// Controls whether this output stream is a "raw stream" or simply a normal
    /// stream.
    ///
    /// # Error
    ///
    /// This function will return an error if the output stream is not actually
    /// connected to a TTY instance, or if querying the TTY instance fails.
    pub fn set_raw(&mut self, raw: bool) -> IoResult<()> {
        match self.inner {
            TTY(ref mut tty) => {
                tty.set_raw(raw).map_err(IoError::from_rtio_error)
            }
            File(..) => {
                Err(IoError {
                    kind: OtherIoError,
                    desc: "stream is not a tty",
                    detail: None,
                })
            }
        }
    }

    /// Returns whether this stream is attached to a TTY instance or not.
    pub fn isatty(&self) -> bool {
        match self.inner {
            TTY(..) => true,
            File(..) => false,
        }
    }
}

impl Writer for StdWriter {
    fn write(&mut self, buf: &[u8]) -> IoResult<()> {
        // As with stdin on windows, stdout often can't handle writes of large
        // sizes. For an example, see #14940. For this reason, chunk the output
        // buffer on windows, but on unix we can just write the whole buffer all
        // at once.
        let max_size = if cfg!(windows) {64 * 1024} else {uint::MAX};
        for chunk in buf.chunks(max_size) {
            try!(match self.inner {
                TTY(ref mut tty) => tty.write(chunk),
                File(ref mut file) => file.write(chunk),
            }.map_err(IoError::from_rtio_error))
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    iotest!(fn smoke() {
        // Just make sure we can acquire handles
        stdin();
        stdout();
        stderr();
    })

    iotest!(fn capture_stdout() {
        use io::{ChanReader, ChanWriter};

        let (tx, rx) = channel();
        let (mut r, w) = (ChanReader::new(rx), ChanWriter::new(tx));
        spawn(proc() {
            set_stdout(box w);
            println!("hello!");
        });
        assert_eq!(r.read_to_str().unwrap(), "hello!\n".to_string());
    })

    iotest!(fn capture_stderr() {
        use realstd::comm::channel;
        use realstd::io::{Writer, ChanReader, ChanWriter, Reader};

        let (tx, rx) = channel();
        let (mut r, w) = (ChanReader::new(rx), ChanWriter::new(tx));
        spawn(proc() {
            ::realstd::io::stdio::set_stderr(box w);
            fail!("my special message");
        });
        let s = r.read_to_str().unwrap();
        assert!(s.as_slice().contains("my special message"));
    })
}
