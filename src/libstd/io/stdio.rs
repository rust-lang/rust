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

use cell::RefCell;
use cmp;
use fmt;
use io::lazy::Lazy;
use io::{self, BufReader, LineWriter};
use sync::{Arc, Mutex, MutexGuard};
use sys::stdio;

/// Stdout used by print! and println! macroses
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

/// Construct a new raw handle to the standard input of this process.
///
/// The returned handle does not interact with any other handles created nor
/// handles returned by `std::io::stdin`. Data buffered by the `std::io::stdin`
/// handles is **not** available to raw handles returned from this function.
///
/// The returned handle has no external synchronization or buffering.
fn stdin_raw() -> StdinRaw { StdinRaw(stdio::Stdin::new()) }

/// Construct a new raw handle to the standard input stream of this process.
///
/// The returned handle does not interact with any other handles created nor
/// handles returned by `std::io::stdout`. Note that data is buffered by the
/// `std::io::stdin` handles so writes which happen via this raw handle may
/// appear before previous writes.
///
/// The returned handle has no external synchronization or buffering layered on
/// top.
fn stdout_raw() -> StdoutRaw { StdoutRaw(stdio::Stdout::new()) }

/// Construct a new raw handle to the standard input stream of this process.
///
/// The returned handle does not interact with any other handles created nor
/// handles returned by `std::io::stdout`.
///
/// The returned handle has no external synchronization or buffering layered on
/// top.
fn stderr_raw() -> StderrRaw { StderrRaw(stdio::Stderr::new()) }

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

/// A handle to the standard input stream of a process.
///
/// Each handle is a shared reference to a global buffer of input data to this
/// process. A handle can be `lock`'d to gain full access to `BufRead` methods
/// (e.g. `.lines()`). Writes to this handle are otherwise locked with respect
/// to other writes.
///
/// This handle implements the `Read` trait, but beware that concurrent reads
/// of `Stdin` must be executed with care.
#[stable(feature = "rust1", since = "1.0.0")]
pub struct Stdin {
    inner: Arc<Mutex<BufReader<StdinRaw>>>,
}

/// A locked reference to the a `Stdin` handle.
///
/// This handle implements both the `Read` and `BufRead` traits and is
/// constructed via the `lock` method on `Stdin`.
#[stable(feature = "rust1", since = "1.0.0")]
pub struct StdinLock<'a> {
    inner: MutexGuard<'a, BufReader<StdinRaw>>,
}

/// Create a new handle to the global standard input stream of this process.
///
/// The handle returned refers to a globally shared buffer between all threads.
/// Access is synchronized and can be explicitly controlled with the `lock()`
/// method.
///
/// The `Read` trait is implemented for the returned value but the `BufRead`
/// trait is not due to the global nature of the standard input stream. The
/// locked version, `StdinLock`, implements both `Read` and `BufRead`, however.
#[stable(feature = "rust1", since = "1.0.0")]
pub fn stdin() -> Stdin {
    static INSTANCE: Lazy<Mutex<BufReader<StdinRaw>>> = lazy_init!(stdin_init);
    return Stdin {
        inner: INSTANCE.get().expect("cannot access stdin during shutdown"),
    };

    fn stdin_init() -> Arc<Mutex<BufReader<StdinRaw>>> {
        // The default buffer capacity is 64k, but apparently windows
        // doesn't like 64k reads on stdin. See #13304 for details, but the
        // idea is that on windows we use a slightly smaller buffer that's
        // been seen to be acceptable.
        Arc::new(Mutex::new(if cfg!(windows) {
            BufReader::with_capacity(8 * 1024, stdin_raw())
        } else {
            BufReader::new(stdin_raw())
        }))
    }
}

impl Stdin {
    /// Lock this handle to the standard input stream, returning a readable
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
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn read_line(&mut self, buf: &mut String) -> io::Result<usize> {
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
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<'a> Read for StdinLock<'a> {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        self.inner.read(buf)
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
#[stable(feature = "rust1", since = "1.0.0")]
pub struct Stdout {
    // FIXME: this should be LineWriter or BufWriter depending on the state of
    //        stdout (tty or not). Note that if this is not line buffered it
    //        should also flush-on-panic or some form of flush-on-abort.
    inner: Arc<Mutex<LineWriter<StdoutRaw>>>,
}

/// A locked reference to the a `Stdout` handle.
///
/// This handle implements the `Write` trait and is constructed via the `lock`
/// method on `Stdout`.
#[stable(feature = "rust1", since = "1.0.0")]
pub struct StdoutLock<'a> {
    inner: MutexGuard<'a, LineWriter<StdoutRaw>>,
}

/// Constructs a new reference to the standard output of the current process.
///
/// Each handle returned is a reference to a shared global buffer whose access
/// is synchronized via a mutex. Explicit control over synchronization is
/// provided via the `lock` method.
///
/// The returned handle implements the `Write` trait.
#[stable(feature = "rust1", since = "1.0.0")]
pub fn stdout() -> Stdout {
    static INSTANCE: Lazy<Mutex<LineWriter<StdoutRaw>>> = lazy_init!(stdout_init);
    return Stdout {
        inner: INSTANCE.get().expect("cannot access stdout during shutdown"),
    };

    fn stdout_init() -> Arc<Mutex<LineWriter<StdoutRaw>>> {
        Arc::new(Mutex::new(LineWriter::new(stdout_raw())))
    }
}

impl Stdout {
    /// Lock this handle to the standard output stream, returning a writable
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
    fn write_fmt(&mut self, fmt: fmt::Arguments) -> io::Result<()> {
        self.lock().write_fmt(fmt)
    }
}
#[stable(feature = "rust1", since = "1.0.0")]
impl<'a> Write for StdoutLock<'a> {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        self.inner.write(&buf[..cmp::min(buf.len(), OUT_MAX)])
    }
    fn flush(&mut self) -> io::Result<()> { self.inner.flush() }
}

/// A handle to the standard error stream of a process.
///
/// For more information, see `stderr`
#[stable(feature = "rust1", since = "1.0.0")]
pub struct Stderr {
    inner: Arc<Mutex<StderrRaw>>,
}

/// A locked reference to the a `Stderr` handle.
///
/// This handle implements the `Write` trait and is constructed via the `lock`
/// method on `Stderr`.
#[stable(feature = "rust1", since = "1.0.0")]
pub struct StderrLock<'a> {
    inner: MutexGuard<'a, StderrRaw>,
}

/// Constructs a new reference to the standard error stream of a process.
///
/// Each returned handle is synchronized amongst all other handles created from
/// this function. No handles are buffered, however.
///
/// The returned handle implements the `Write` trait.
#[stable(feature = "rust1", since = "1.0.0")]
pub fn stderr() -> Stderr {
    static INSTANCE: Lazy<Mutex<StderrRaw>> = lazy_init!(stderr_init);
    return Stderr {
        inner: INSTANCE.get().expect("cannot access stderr during shutdown"),
    };

    fn stderr_init() -> Arc<Mutex<StderrRaw>> {
        Arc::new(Mutex::new(stderr_raw()))
    }
}

impl Stderr {
    /// Lock this handle to the standard error stream, returning a writable
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
    fn write_fmt(&mut self, fmt: fmt::Arguments) -> io::Result<()> {
        self.lock().write_fmt(fmt)
    }
}
#[stable(feature = "rust1", since = "1.0.0")]
impl<'a> Write for StderrLock<'a> {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        self.inner.write(&buf[..cmp::min(buf.len(), OUT_MAX)])
    }
    fn flush(&mut self) -> io::Result<()> { self.inner.flush() }
}

/// Resets the task-local stderr handle to the specified writer
///
/// This will replace the current task's stderr handle, returning the old
/// handle. All future calls to `panic!` and friends will emit their output to
/// this specified handle.
///
/// Note that this does not need to be called for all new tasks; the default
/// output handle is to the process's stderr stream.
#[unstable(feature = "set_stdio",
           reason = "this function may disappear completely or be replaced \
                     with a more general mechanism")]
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

/// Resets the task-local stdout handle to the specified writer
///
/// This will replace the current task's stdout handle, returning the old
/// handle. All future calls to `print!` and friends will emit their output to
/// this specified handle.
///
/// Note that this does not need to be called for all new tasks; the default
/// output handle is to the process's stdout stream.
#[unstable(feature = "set_stdio",
           reason = "this function may disappear completely or be replaced \
                     with a more general mechanism")]
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
           reason = "implementation detail which may disappear or be replaced at any time")]
#[doc(hidden)]
pub fn _print(args: fmt::Arguments) {
    if let Err(e) = LOCAL_STDOUT.with(|s| match s.borrow_mut().as_mut() {
        Some(w) => w.write_fmt(args),
        None => stdout().write_fmt(args)
    }) {
        panic!("failed printing to stdout: {}", e);
    }
}

#[cfg(test)]
mod test {
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
