// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

/*!

This modules provides bindings to the local event loop's TTY interface, using it
to have synchronous, but non-blocking versions of stdio. These handles can be
inspected for information about terminal dimensions or related information
about the stream or terminal that it is attached to.

# Example

```rust
use std::io;

let mut out = io::stdout();
out.write(bytes!("Hello, world!"));
```

*/

use fmt;
use io::buffered::LineBufferedWriter;
use io::{Reader, Writer, io_error, IoError, OtherIoError,
         standard_error, EndOfFile};
use libc;
use option::{Option, Some, None};
use prelude::drop;
use result::{Ok, Err};
use rt::local::Local;
use rt::rtio::{DontClose, IoFactory, LocalIo, RtioFileStream, RtioTTY};
use rt::task::Task;
use util;

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
    TTY(~RtioTTY),
    File(~RtioFileStream),
}

fn src<T>(fd: libc::c_int, readable: bool, f: |StdSource| -> T) -> T {
    LocalIo::maybe_raise(|io| {
        Ok(match io.tty_open(fd, readable) {
            Ok(tty) => f(TTY(tty)),
            Err(_) => f(File(io.fs_from_raw_fd(fd, DontClose))),
        })
    }).unwrap()
}

/// Creates a new non-blocking handle to the stdin of the current process.
///
/// See `stdout()` for notes about this function.
pub fn stdin() -> StdReader {
    src(libc::STDIN_FILENO, true, |src| StdReader { inner: src })
}

/// Creates a new non-blocking handle to the stdout of the current process.
///
/// Note that this is a fairly expensive operation in that at least one memory
/// allocation is performed. Additionally, this must be called from a runtime
/// task context because the stream returned will be a non-blocking object using
/// the local scheduler to perform the I/O.
pub fn stdout() -> StdWriter {
    src(libc::STDOUT_FILENO, false, |src| StdWriter { inner: src })
}

/// Creates a new non-blocking handle to the stderr of the current process.
///
/// See `stdout()` for notes about this function.
pub fn stderr() -> StdWriter {
    src(libc::STDERR_FILENO, false, |src| StdWriter { inner: src })
}

fn reset_helper(w: ~Writer,
                f: |&mut Task, ~Writer| -> Option<~Writer>) -> Option<~Writer> {
    let mut t = Local::borrow(None::<Task>);
    // Be sure to flush any pending output from the writer
    match f(t.get(), w) {
        Some(mut w) => {
            drop(t);
            w.flush();
            Some(w)
        }
        None => None
    }
}

/// Resets the task-local stdout handle to the specified writer
///
/// This will replace the current task's stdout handle, returning the old
/// handle. All future calls to `print` and friends will emit their output to
/// this specified handle.
///
/// Note that this does not need to be called for all new tasks; the default
/// output handle is to the process's stdout stream.
pub fn set_stdout(stdout: ~Writer) -> Option<~Writer> {
    reset_helper(stdout, |t, w| util::replace(&mut t.stdout, Some(w)))
}

/// Resets the task-local stderr handle to the specified writer
///
/// This will replace the current task's stderr handle, returning the old
/// handle. Currently, the stderr handle is used for printing failure messages
/// during task failure.
///
/// Note that this does not need to be called for all new tasks; the default
/// output handle is to the process's stderr stream.
pub fn set_stderr(stderr: ~Writer) -> Option<~Writer> {
    reset_helper(stderr, |t, w| util::replace(&mut t.stderr, Some(w)))
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
fn with_task_stdout(f: |&mut Writer|) {
    let task: Option<~Task> = Local::try_take();
    match task {
        Some(mut task) => {
            // Printing may run arbitrary code, so ensure that the task is in
            // TLS to allow all std services. Note that this means a print while
            // printing won't use the task's normal stdout handle, but this is
            // necessary to ensure safety (no aliasing).
            let mut my_stdout = task.stdout.take();
            Local::put(task);

            if my_stdout.is_none() {
                my_stdout = Some(~LineBufferedWriter::new(stdout()) as ~Writer);
            }
            f(*my_stdout.get_mut_ref());

            // Note that we need to be careful when putting the stdout handle
            // back into the task. If the handle was set to `Some` while
            // printing, then we can run aribitrary code when destroying the
            // previous handle. This means that the local task needs to be in
            // TLS while we do this.
            //
            // To protect against this, we do a little dance in which we
            // temporarily take the task, swap the handles, put the task in TLS,
            // and only then drop the previous handle.
            let mut t = Local::borrow(None::<Task>);
            let prev = util::replace(&mut t.get().stdout, my_stdout);
            drop(t);
            drop(prev);
        }

        None => {
            struct Stdout;
            impl Writer for Stdout {
                fn write(&mut self, data: &[u8]) {
                    unsafe {
                        libc::write(libc::STDOUT_FILENO,
                                    data.as_ptr() as *libc::c_void,
                                    data.len() as libc::size_t);
                    }
                }
            }
            let mut io = Stdout;
            f(&mut io as &mut Writer);
        }
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
        io.write(s.as_bytes());
        io.write(['\n' as u8]);
    })
}

/// Similar to `print`, but takes a `fmt::Arguments` structure to be compatible
/// with the `format_args!` macro.
pub fn print_args(fmt: &fmt::Arguments) {
    with_task_stdout(|io| fmt::write(io, fmt))
}

/// Similar to `println`, but takes a `fmt::Arguments` structure to be
/// compatible with the `format_args!` macro.
pub fn println_args(fmt: &fmt::Arguments) {
    with_task_stdout(|io| fmt::writeln(io, fmt))
}

/// Representation of a reader of a standard input stream
pub struct StdReader {
    priv inner: StdSource
}

impl Reader for StdReader {
    fn read(&mut self, buf: &mut [u8]) -> Option<uint> {
        let ret = match self.inner {
            TTY(ref mut tty) => tty.read(buf),
            File(ref mut file) => file.read(buf).map(|i| i as uint),
        };
        match ret {
            // When reading a piped stdin, libuv will return 0-length reads when
            // stdin reaches EOF. For pretty much all other streams it will
            // return an actual EOF error, but apparently for stdin it's a
            // little different. Hence, here we convert a 0 length read to an
            // end-of-file indicator so the caller knows to stop reading.
            Ok(0) => {
                io_error::cond.raise(standard_error(EndOfFile));
                None
            }
            Ok(amt) => Some(amt),
            Err(e) => {
                io_error::cond.raise(e);
                None
            }
        }
    }

    fn eof(&mut self) -> bool { false }
}

/// Representation of a writer to a standard output stream
pub struct StdWriter {
    priv inner: StdSource
}

impl StdWriter {
    /// Gets the size of this output window, if possible. This is typically used
    /// when the writer is attached to something like a terminal, this is used
    /// to fetch the dimensions of the terminal.
    ///
    /// If successful, returns Some((width, height)).
    ///
    /// # Failure
    ///
    /// This function will raise on the `io_error` condition if an error
    /// happens.
    pub fn winsize(&mut self) -> Option<(int, int)> {
        match self.inner {
            TTY(ref mut tty) => {
                match tty.get_winsize() {
                    Ok(p) => Some(p),
                    Err(e) => {
                        io_error::cond.raise(e);
                        None
                    }
                }
            }
            File(..) => {
                io_error::cond.raise(IoError {
                    kind: OtherIoError,
                    desc: "stream is not a tty",
                    detail: None,
                });
                None
            }
        }
    }

    /// Controls whether this output stream is a "raw stream" or simply a normal
    /// stream.
    ///
    /// # Failure
    ///
    /// This function will raise on the `io_error` condition if an error
    /// happens.
    pub fn set_raw(&mut self, raw: bool) {
        match self.inner {
            TTY(ref mut tty) => {
                match tty.set_raw(raw) {
                    Ok(()) => {},
                    Err(e) => io_error::cond.raise(e),
                }
            }
            File(..) => {
                io_error::cond.raise(IoError {
                    kind: OtherIoError,
                    desc: "stream is not a tty",
                    detail: None,
                });
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
    fn write(&mut self, buf: &[u8]) {
        let ret = match self.inner {
            TTY(ref mut tty) => tty.write(buf),
            File(ref mut file) => file.write(buf),
        };
        match ret {
            Ok(()) => {}
            Err(e) => io_error::cond.raise(e)
        }
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
        use io::comm_adapters::{PortReader, ChanWriter};

        let (p, c) = Chan::new();
        let (mut r, w) = (PortReader::new(p), ChanWriter::new(c));
        do spawn {
            set_stdout(~w as ~Writer);
            println!("hello!");
        }
        assert_eq!(r.read_to_str(), ~"hello!\n");
    })

    iotest!(fn capture_stderr() {
        use io::comm_adapters::{PortReader, ChanWriter};

        let (p, c) = Chan::new();
        let (mut r, w) = (PortReader::new(p), ChanWriter::new(c));
        do spawn {
            set_stderr(~w as ~Writer);
            fail!("my special message");
        }
        let s = r.read_to_str();
        assert!(s.contains("my special message"));
    })
}
