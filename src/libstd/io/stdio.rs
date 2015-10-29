// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use prelude::v1::*;
use io::prelude::*;

use cell::{RefCell, BorrowState};
use cmp;
use fmt;
use io::lazy::Lazy;
use io::{self, BufReader, LineWriter};
use sync::{Arc, Mutex, MutexGuard};
use sys::stdio;
use sys_common::io::{read_to_end_uninitialized};
use sys_common::remutex::{ReentrantMutex, ReentrantMutexGuard};
use libc;

/// Stdout used by print! and println! macros
thread_local! {
    static LOCAL_STDOUT: RefCell<Option<Box<Write + Send>>> = {
        RefCell::new(None)
    }
}

/// A handle to a raw instance of the standard input stream of this process.
///
/// This handle is not synchronized or buffered in any fashion. Constructed via
/// the `std::io::stdio::stdin_raw` function.
struct StdinRaw(stdio::Stdin);

/// A handle to a raw instance of the standard output stream of this process.
///
/// This handle is not synchronized or buffered in any fashion. Constructed via
/// the `std::io::stdio::stdout_raw` function.
struct StdoutRaw(stdio::Stdout);

/// A handle to a raw instance of the standard output stream of this process.
///
/// This handle is not synchronized or buffered in any fashion. Constructed via
/// the `std::io::stdio::stderr_raw` function.
struct StderrRaw(stdio::Stderr);

/// Constructs a new raw handle to the standard input of this process.
///
/// The returned handle does not interact with any other handles created nor
/// handles returned by `std::io::stdin`. Data buffered by the `std::io::stdin`
/// handles is **not** available to raw handles returned from this function.
///
/// The returned handle has no external synchronization or buffering.
fn stdin_raw() -> io::Result<StdinRaw> { stdio::Stdin::new().map(StdinRaw) }

/// Constructs a new raw handle to the standard output stream of this process.
///
/// The returned handle does not interact with any other handles created nor
/// handles returned by `std::io::stdout`. Note that data is buffered by the
/// `std::io::stdout` handles so writes which happen via this raw handle may
/// appear before previous writes.
///
/// The returned handle has no external synchronization or buffering layered on
/// top.
fn stdout_raw() -> io::Result<StdoutRaw> { stdio::Stdout::new().map(StdoutRaw) }

/// Constructs a new raw handle to the standard error stream of this process.
///
/// The returned handle does not interact with any other handles created nor
/// handles returned by `std::io::stderr`.
///
/// The returned handle has no external synchronization or buffering layered on
/// top.
fn stderr_raw() -> io::Result<StderrRaw> { stdio::Stderr::new().map(StderrRaw) }

impl Read for StdinRaw {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> { self.0.read(buf) }
}
impl Write for StdoutRaw {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> { self.0.write(buf) }
    fn flush(&mut self) -> io::Result<()> { Ok(()) }
}
impl Write for StderrRaw {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> { self.0.write(buf) }
    fn flush(&mut self) -> io::Result<()> { Ok(()) }
}

enum Maybe<T> {
    Real(T),
    Fake,
}

impl<W: io::Write> io::Write for Maybe<W> {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        match *self {
            Maybe::Real(ref mut w) => handle_ebadf(w.write(buf), buf.len()),
            Maybe::Fake => Ok(buf.len())
        }
    }

    fn flush(&mut self) -> io::Result<()> {
        match *self {
            Maybe::Real(ref mut w) => handle_ebadf(w.flush(), ()),
            Maybe::Fake => Ok(())
        }
    }
}

impl<R: io::Read> io::Read for Maybe<R> {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        match *self {
            Maybe::Real(ref mut r) => handle_ebadf(r.read(buf), buf.len()),
            Maybe::Fake => Ok(0)
        }
    }
}

fn handle_ebadf<T>(r: io::Result<T>, default: T) -> io::Result<T> {
    #[cfg(windows)]
    const ERR: libc::c_int = libc::ERROR_INVALID_HANDLE;
    #[cfg(not(windows))]
    const ERR: libc::c_int = libc::EBADF;

    match r {
        Err(ref e) if e.raw_os_error() == Some(ERR) => Ok(default),
        r => r
    }
}

/// A handle to the standard input stream of a process.
///
/// Each handle is a shared reference to a global buffer of input data to this
/// process. A handle can be `lock`'d to gain full access to `BufRead` methods
/// (e.g. `.lines()`). Writes to this handle are otherwise locked with respect
/// to other writes.
///
/// This handle implements the `Read` trait, but beware that concurrent reads
/// of `Stdin` must be executed with care.
///
/// Created by the function `io::stdin()`.
#[stable(feature = "rust1", since = "1.0.0")]
pub struct Stdin {
    inner: Arc<Mutex<BufReader<Maybe<StdinRaw>>>>,
}

/// A locked reference to the `Stdin` handle.
///
/// This handle implements both the `Read` and `BufRead` traits and is
/// constructed via the `lock` method on `Stdin`.
#[stable(feature = "rust1", since = "1.0.0")]
pub struct StdinLock<'a> {
    inner: MutexGuard<'a, BufReader<Maybe<StdinRaw>>>,
}

/// Constructs a new handle to the standard input of the current process.
///
/// Each handle returned is a reference to a shared global buffer whose access
/// is synchronized via a mutex. If you need more explicit control over
/// locking, see the [lock() method][lock].
///
/// [lock]: struct.Stdin.html#method.lock
///
/// # Examples
///
/// Using implicit synchronization:
///
/// ```
/// use std::io::{self, Read};
///
/// # fn foo() -> io::Result<String> {
/// let mut buffer = String::new();
/// try!(io::stdin().read_to_string(&mut buffer));
/// # Ok(buffer)
/// # }
/// ```
///
/// Using explicit synchronization:
///
/// ```
/// use std::io::{self, Read};
///
/// # fn foo() -> io::Result<String> {
/// let mut buffer = String::new();
/// let stdin = io::stdin();
/// let mut handle = stdin.lock();
///
/// try!(handle.read_to_string(&mut buffer));
/// # Ok(buffer)
/// # }
/// ```
#[stable(feature = "rust1", since = "1.0.0")]
pub fn stdin() -> Stdin {
    static INSTANCE: Lazy<Mutex<BufReader<Maybe<StdinRaw>>>> = Lazy::new(stdin_init);
    return Stdin {
        inner: INSTANCE.get().expect("cannot access stdin during shutdown"),
    };

    fn stdin_init() -> Arc<Mutex<BufReader<Maybe<StdinRaw>>>> {
        let stdin = match stdin_raw() {
            Ok(stdin) => Maybe::Real(stdin),
            _ => Maybe::Fake
        };

        // The default buffer capacity is 64k, but apparently windows
        // doesn't like 64k reads on stdin. See #13304 for details, but the
        // idea is that on windows we use a slightly smaller buffer that's
        // been seen to be acceptable.
        Arc::new(Mutex::new(if cfg!(windows) {
            BufReader::with_capacity(8 * 1024, stdin)
        } else {
            BufReader::new(stdin)
        }))
    }
}

impl Stdin {
    /// Locks this handle to the standard input stream, returning a readable
    /// guard.
    ///
    /// The lock is released when the returned lock goes out of scope. The
    /// returned guard also implements the `Read` and `BufRead` traits for
    /// accessing the underlying data.
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn lock(&self) -> StdinLock {
        StdinLock { inner: self.inner.lock().unwrap_or_else(|e| e.into_inner()) }
    }

    /// Locks this handle and reads a line of input into the specified buffer.
    ///
    /// For detailed semantics of this method, see the documentation on
    /// `BufRead::read_line`.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use std::io;
    ///
    /// let mut input = String::new();
    /// match io::stdin().read_line(&mut input) {
    ///     Ok(n) => {
    ///         println!("{} bytes read", n);
    ///         println!("{}", input);
    ///     }
    ///     Err(error) => println!("error: {}", error),
    /// }
    /// ```
    ///
    /// You can run the example one of two ways:
    ///
    /// - Pipe some text to it, e.g. `printf foo | path/to/executable`
    /// - Give it text interactively by running the executable directly,
    //    in which case it will wait for the Enter key to be pressed before
    ///   continuing
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn read_line(&self, buf: &mut String) -> io::Result<usize> {
        self.lock().read_line(buf)
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl Read for Stdin {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        self.lock().read(buf)
    }
    fn read_to_end(&mut self, buf: &mut Vec<u8>) -> io::Result<usize> {
        self.lock().read_to_end(buf)
    }
    fn read_to_string(&mut self, buf: &mut String) -> io::Result<usize> {
        self.lock().read_to_string(buf)
    }
    fn read_exact(&mut self, buf: &mut [u8]) -> io::Result<()> {
        self.lock().read_exact(buf)
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<'a> Read for StdinLock<'a> {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        self.inner.read(buf)
    }
    fn read_to_end(&mut self, buf: &mut Vec<u8>) -> io::Result<usize> {
        unsafe { read_to_end_uninitialized(self, buf) }
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<'a> BufRead for StdinLock<'a> {
    fn fill_buf(&mut self) -> io::Result<&[u8]> { self.inner.fill_buf() }
    fn consume(&mut self, n: usize) { self.inner.consume(n) }
}

// As with stdin on windows, stdout often can't handle writes of large
// sizes. For an example, see #14940. For this reason, don't try to
// write the entire output buffer on windows. On unix we can just
// write the whole buffer all at once.
//
// For some other references, it appears that this problem has been
// encountered by others [1] [2]. We choose the number 8KB just because
// libuv does the same.
//
// [1]: https://tahoe-lafs.org/trac/tahoe-lafs/ticket/1232
// [2]: http://www.mail-archive.com/log4net-dev@logging.apache.org/msg00661.html
#[cfg(windows)]
const OUT_MAX: usize = 8192;
#[cfg(unix)]
const OUT_MAX: usize = ::usize::MAX;

/// A handle to the global standard output stream of the current process.
///
/// Each handle shares a global buffer of data to be written to the standard
/// output stream. Access is also synchronized via a lock and explicit control
/// over locking is available via the `lock` method.
///
/// Created by the function `io::stdout()`.
#[stable(feature = "rust1", since = "1.0.0")]
pub struct Stdout {
    // FIXME: this should be LineWriter or BufWriter depending on the state of
    //        stdout (tty or not). Note that if this is not line buffered it
    //        should also flush-on-panic or some form of flush-on-abort.
    inner: Arc<ReentrantMutex<RefCell<LineWriter<Maybe<StdoutRaw>>>>>,
}

/// A locked reference to the `Stdout` handle.
///
/// This handle implements the `Write` trait and is constructed via the `lock`
/// method on `Stdout`.
#[stable(feature = "rust1", since = "1.0.0")]
pub struct StdoutLock<'a> {
    inner: ReentrantMutexGuard<'a, RefCell<LineWriter<Maybe<StdoutRaw>>>>,
}

/// Constructs a new handle to the standard output of the current process.
///
/// Each handle returned is a reference to a shared global buffer whose access
/// is synchronized via a mutex. If you need more explicit control over
/// locking, see the [lock() method][lock].
///
/// [lock]: struct.Stdout.html#method.lock
///
/// # Examples
///
/// Using implicit synchronization:
///
/// ```
/// use std::io::{self, Write};
///
/// # fn foo() -> io::Result<()> {
/// try!(io::stdout().write(b"hello world"));
///
/// # Ok(())
/// # }
/// ```
///
/// Using explicit synchronization:
///
/// ```
/// use std::io::{self, Write};
///
/// # fn foo() -> io::Result<()> {
/// let stdout = io::stdout();
/// let mut handle = stdout.lock();
///
/// try!(handle.write(b"hello world"));
///
/// # Ok(())
/// # }
/// ```
#[stable(feature = "rust1", since = "1.0.0")]
pub fn stdout() -> Stdout {
    static INSTANCE: Lazy<ReentrantMutex<RefCell<LineWriter<Maybe<StdoutRaw>>>>>
        = Lazy::new(stdout_init);
    return Stdout {
        inner: INSTANCE.get().expect("cannot access stdout during shutdown"),
    };

    fn stdout_init() -> Arc<ReentrantMutex<RefCell<LineWriter<Maybe<StdoutRaw>>>>> {
        let stdout = match stdout_raw() {
            Ok(stdout) => Maybe::Real(stdout),
            _ => Maybe::Fake,
        };
        Arc::new(ReentrantMutex::new(RefCell::new(LineWriter::new(stdout))))
    }
}

impl Stdout {
    /// Locks this handle to the standard output stream, returning a writable
    /// guard.
    ///
    /// The lock is released when the returned lock goes out of scope. The
    /// returned guard also implements the `Write` trait for writing data.
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn lock(&self) -> StdoutLock {
        StdoutLock { inner: self.inner.lock().unwrap_or_else(|e| e.into_inner()) }
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl Write for Stdout {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        self.lock().write(buf)
    }
    fn flush(&mut self) -> io::Result<()> {
        self.lock().flush()
    }
    fn write_all(&mut self, buf: &[u8]) -> io::Result<()> {
        self.lock().write_all(buf)
    }
    fn write_fmt(&mut self, args: fmt::Arguments) -> io::Result<()> {
        self.lock().write_fmt(args)
    }
}
#[stable(feature = "rust1", since = "1.0.0")]
impl<'a> Write for StdoutLock<'a> {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        self.inner.borrow_mut().write(&buf[..cmp::min(buf.len(), OUT_MAX)])
    }
    fn flush(&mut self) -> io::Result<()> {
        self.inner.borrow_mut().flush()
    }
}

/// A handle to the standard error stream of a process.
///
/// For more information, see `stderr`
#[stable(feature = "rust1", since = "1.0.0")]
pub struct Stderr {
    inner: Arc<ReentrantMutex<RefCell<Maybe<StderrRaw>>>>,
}

/// A locked reference to the `Stderr` handle.
///
/// This handle implements the `Write` trait and is constructed via the `lock`
/// method on `Stderr`.
#[stable(feature = "rust1", since = "1.0.0")]
pub struct StderrLock<'a> {
    inner: ReentrantMutexGuard<'a, RefCell<Maybe<StderrRaw>>>,
}

/// Constructs a new handle to the standard error of the current process.
///
/// This handle is not buffered.
///
/// # Examples
///
/// Using implicit synchronization:
///
/// ```
/// use std::io::{self, Write};
///
/// # fn foo() -> io::Result<()> {
/// try!(io::stderr().write(b"hello world"));
///
/// # Ok(())
/// # }
/// ```
///
/// Using explicit synchronization:
///
/// ```
/// use std::io::{self, Write};
///
/// # fn foo() -> io::Result<()> {
/// let stderr = io::stderr();
/// let mut handle = stderr.lock();
///
/// try!(handle.write(b"hello world"));
///
/// # Ok(())
/// # }
/// ```
#[stable(feature = "rust1", since = "1.0.0")]
pub fn stderr() -> Stderr {
    static INSTANCE: Lazy<ReentrantMutex<RefCell<Maybe<StderrRaw>>>> = Lazy::new(stderr_init);
    return Stderr {
        inner: INSTANCE.get().expect("cannot access stderr during shutdown"),
    };

    fn stderr_init() -> Arc<ReentrantMutex<RefCell<Maybe<StderrRaw>>>> {
        let stderr = match stderr_raw() {
            Ok(stderr) => Maybe::Real(stderr),
            _ => Maybe::Fake,
        };
        Arc::new(ReentrantMutex::new(RefCell::new(stderr)))
    }
}

impl Stderr {
    /// Locks this handle to the standard error stream, returning a writable
    /// guard.
    ///
    /// The lock is released when the returned lock goes out of scope. The
    /// returned guard also implements the `Write` trait for writing data.
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn lock(&self) -> StderrLock {
        StderrLock { inner: self.inner.lock().unwrap_or_else(|e| e.into_inner()) }
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl Write for Stderr {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        self.lock().write(buf)
    }
    fn flush(&mut self) -> io::Result<()> {
        self.lock().flush()
    }
    fn write_all(&mut self, buf: &[u8]) -> io::Result<()> {
        self.lock().write_all(buf)
    }
    fn write_fmt(&mut self, args: fmt::Arguments) -> io::Result<()> {
        self.lock().write_fmt(args)
    }
}
#[stable(feature = "rust1", since = "1.0.0")]
impl<'a> Write for StderrLock<'a> {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        self.inner.borrow_mut().write(&buf[..cmp::min(buf.len(), OUT_MAX)])
    }
    fn flush(&mut self) -> io::Result<()> {
        self.inner.borrow_mut().flush()
    }
}

/// Resets the thread-local stderr handle to the specified writer
///
/// This will replace the current thread's stderr handle, returning the old
/// handle. All future calls to `panic!` and friends will emit their output to
/// this specified handle.
///
/// Note that this does not need to be called for all new threads; the default
/// output handle is to the process's stderr stream.
#[unstable(feature = "set_stdio",
           reason = "this function may disappear completely or be replaced \
                     with a more general mechanism",
           issue = "0")]
#[doc(hidden)]
pub fn set_panic(sink: Box<Write + Send>) -> Option<Box<Write + Send>> {
    use panicking::LOCAL_STDERR;
    use mem;
    LOCAL_STDERR.with(move |slot| {
        mem::replace(&mut *slot.borrow_mut(), Some(sink))
    }).and_then(|mut s| {
        let _ = s.flush();
        Some(s)
    })
}

/// Resets the thread-local stdout handle to the specified writer
///
/// This will replace the current thread's stdout handle, returning the old
/// handle. All future calls to `print!` and friends will emit their output to
/// this specified handle.
///
/// Note that this does not need to be called for all new threads; the default
/// output handle is to the process's stdout stream.
#[unstable(feature = "set_stdio",
           reason = "this function may disappear completely or be replaced \
                     with a more general mechanism",
           issue = "0")]
#[doc(hidden)]
pub fn set_print(sink: Box<Write + Send>) -> Option<Box<Write + Send>> {
    use mem;
    LOCAL_STDOUT.with(move |slot| {
        mem::replace(&mut *slot.borrow_mut(), Some(sink))
    }).and_then(|mut s| {
        let _ = s.flush();
        Some(s)
    })
}

#[unstable(feature = "print",
           reason = "implementation detail which may disappear or be replaced at any time",
           issue = "0")]
#[doc(hidden)]
pub fn _print(args: fmt::Arguments) {
    let result = _try_print(args);
    if let Err(e) = result {
        panic!("failed printing to stdout: {}", e);
    }
}

#[unstable(feature = "print",
           reason = "implementation detail which may disappear or be replaced at any time",
           issue = "0")]
#[doc(hidden)]
pub fn _try_print(args: fmt::Arguments) -> io::Result<()> {
    LOCAL_STDOUT.with(|s| {
        if s.borrow_state() == BorrowState::Unused {
            if let Some(w) = s.borrow_mut().as_mut() {
                return w.write_fmt(args);
            }
        }
        stdout().write_fmt(args)
    })
}

#[cfg(test)]
mod tests {
    use thread;
    use super::*;

    #[test]
    fn panic_doesnt_poison() {
        thread::spawn(|| {
            let _a = stdin();
            let _a = _a.lock();
            let _a = stdout();
            let _a = _a.lock();
            let _a = stderr();
            let _a = _a.lock();
            panic!();
        }).join().unwrap_err();

        let _a = stdin();
        let _a = _a.lock();
        let _a = stdout();
        let _a = _a.lock();
        let _a = stderr();
        let _a = _a.lock();
    }
}
