// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use io::prelude::*;

use cell::RefCell;
use fmt;
use io::lazy::Lazy;
use io::{self, BufReader, LineWriter};
use sync::{Arc, Mutex, MutexGuard};
use sys::stdio;
use sys_common::remutex::{ReentrantMutex, ReentrantMutexGuard};
use thread::LocalKeyState;

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
    fn read_to_end(&mut self, buf: &mut Vec<u8>) -> io::Result<usize> {
        self.0.read_to_end(buf)
    }
}
impl Write for StdoutRaw {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> { self.0.write(buf) }
    fn flush(&mut self) -> io::Result<()> { self.0.flush() }
}
impl Write for StderrRaw {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> { self.0.write(buf) }
    fn flush(&mut self) -> io::Result<()> { self.0.flush() }
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
            Maybe::Real(ref mut r) => handle_ebadf(r.read(buf), 0),
            Maybe::Fake => Ok(0)
        }
    }
    fn read_to_end(&mut self, buf: &mut Vec<u8>) -> io::Result<usize> {
        match *self {
            Maybe::Real(ref mut r) => handle_ebadf(r.read_to_end(buf), 0),
            Maybe::Fake => Ok(0)
        }
    }
}

fn handle_ebadf<T>(r: io::Result<T>, default: T) -> io::Result<T> {
    use sys::stdio::EBADF_ERR;

    match r {
        Err(ref e) if e.raw_os_error() == Some(EBADF_ERR) => Ok(default),
        r => r
    }
}

/// A handle to the standard input stream of a process.
///
/// Each handle is a shared reference to a global buffer of input data to this
/// process. A handle can be `lock`'d to gain full access to [`BufRead`] methods
/// (e.g. `.lines()`). Reads to this handle are otherwise locked with respect
/// to other reads.
///
/// This handle implements the `Read` trait, but beware that concurrent reads
/// of `Stdin` must be executed with care.
///
/// Created by the [`io::stdin`] method.
///
/// [`io::stdin`]: fn.stdin.html
/// [`BufRead`]: trait.BufRead.html
#[stable(feature = "rust1", since = "1.0.0")]
pub struct Stdin {
    inner: Arc<Mutex<BufReader<Maybe<StdinRaw>>>>,
}

/// A locked reference to the `Stdin` handle.
///
/// This handle implements both the [`Read`] and [`BufRead`] traits, and
/// is constructed via the [`Stdin::lock`] method.
///
/// [`Read`]: trait.Read.html
/// [`BufRead`]: trait.BufRead.html
/// [`Stdin::lock`]: struct.Stdin.html#method.lock
#[stable(feature = "rust1", since = "1.0.0")]
pub struct StdinLock<'a> {
    inner: MutexGuard<'a, BufReader<Maybe<StdinRaw>>>,
}

/// Constructs a new handle to the standard input of the current process.
///
/// Each handle returned is a reference to a shared global buffer whose access
/// is synchronized via a mutex. If you need more explicit control over
/// locking, see the [`lock() method`][lock].
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
/// io::stdin().read_to_string(&mut buffer)?;
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
/// handle.read_to_string(&mut buffer)?;
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

        Arc::new(Mutex::new(BufReader::with_capacity(stdio::STDIN_BUF_SIZE, stdin)))
    }
}

impl Stdin {
    /// Locks this handle to the standard input stream, returning a readable
    /// guard.
    ///
    /// The lock is released when the returned lock goes out of scope. The
    /// returned guard also implements the [`Read`] and [`BufRead`] traits for
    /// accessing the underlying data.
    ///
    /// [`Read`]: trait.Read.html
    /// [`BufRead`]: trait.BufRead.html
    ///
    /// # Examples
    ///
    /// ```
    /// use std::io::{self, Read};
    ///
    /// # fn foo() -> io::Result<String> {
    /// let mut buffer = String::new();
    /// let stdin = io::stdin();
    /// let mut handle = stdin.lock();
    ///
    /// handle.read_to_string(&mut buffer)?;
    /// # Ok(buffer)
    /// # }
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn lock(&self) -> StdinLock {
        StdinLock { inner: self.inner.lock().unwrap_or_else(|e| e.into_inner()) }
    }

    /// Locks this handle and reads a line of input into the specified buffer.
    ///
    /// For detailed semantics of this method, see the documentation on
    /// [`BufRead::read_line`].
    ///
    /// [`BufRead::read_line`]: trait.BufRead.html#method.read_line
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
    ///   in which case it will wait for the Enter key to be pressed before
    ///   continuing
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn read_line(&self, buf: &mut String) -> io::Result<usize> {
        self.lock().read_line(buf)
    }
}

#[stable(feature = "std_debug", since = "1.15.0")]
impl fmt::Debug for Stdin {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.pad("Stdin { .. }")
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
        self.inner.read_to_end(buf)
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<'a> BufRead for StdinLock<'a> {
    fn fill_buf(&mut self) -> io::Result<&[u8]> { self.inner.fill_buf() }
    fn consume(&mut self, n: usize) { self.inner.consume(n) }
}

#[stable(feature = "std_debug", since = "1.15.0")]
impl<'a> fmt::Debug for StdinLock<'a> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.pad("StdinLock { .. }")
    }
}

/// A handle to the global standard output stream of the current process.
///
/// Each handle shares a global buffer of data to be written to the standard
/// output stream. Access is also synchronized via a lock and explicit control
/// over locking is available via the [`lock()`] method.
///
/// Created by the [`io::stdout`] method.
///
/// [`lock()`]: #method.lock
/// [`io::stdout`]: fn.stdout.html
#[stable(feature = "rust1", since = "1.0.0")]
pub struct Stdout {
    // FIXME: this should be LineWriter or BufWriter depending on the state of
    //        stdout (tty or not). Note that if this is not line buffered it
    //        should also flush-on-panic or some form of flush-on-abort.
    inner: Arc<ReentrantMutex<RefCell<LineWriter<Maybe<StdoutRaw>>>>>,
}

/// A locked reference to the `Stdout` handle.
///
/// This handle implements the [`Write`] trait, and is constructed via
/// the [`Stdout::lock`] method.
///
/// [`Write`]: trait.Write.html
/// [`Stdout::lock`]: struct.Stdout.html#method.lock
#[stable(feature = "rust1", since = "1.0.0")]
pub struct StdoutLock<'a> {
    inner: ReentrantMutexGuard<'a, RefCell<LineWriter<Maybe<StdoutRaw>>>>,
}

/// Constructs a new handle to the standard output of the current process.
///
/// Each handle returned is a reference to a shared global buffer whose access
/// is synchronized via a mutex. If you need more explicit control over
/// locking, see the [Stdout::lock] method.
///
/// [Stdout::lock]: struct.Stdout.html#method.lock
///
/// # Examples
///
/// Using implicit synchronization:
///
/// ```
/// use std::io::{self, Write};
///
/// # fn foo() -> io::Result<()> {
/// io::stdout().write(b"hello world")?;
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
/// handle.write(b"hello world")?;
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
    ///
    /// # Examples
    ///
    /// ```
    /// use std::io::{self, Write};
    ///
    /// # fn foo() -> io::Result<()> {
    /// let stdout = io::stdout();
    /// let mut handle = stdout.lock();
    ///
    /// handle.write(b"hello world")?;
    ///
    /// # Ok(())
    /// # }
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn lock(&self) -> StdoutLock {
        StdoutLock { inner: self.inner.lock().unwrap_or_else(|e| e.into_inner()) }
    }
}

#[stable(feature = "std_debug", since = "1.15.0")]
impl fmt::Debug for Stdout {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.pad("Stdout { .. }")
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
        self.inner.borrow_mut().write(buf)
    }
    fn flush(&mut self) -> io::Result<()> {
        self.inner.borrow_mut().flush()
    }
}

#[stable(feature = "std_debug", since = "1.15.0")]
impl<'a> fmt::Debug for StdoutLock<'a> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.pad("StdoutLock { .. }")
    }
}

/// A handle to the standard error stream of a process.
///
/// For more information, see the [`io::stderr`] method.
///
/// [`io::stderr`]: fn.stderr.html
#[stable(feature = "rust1", since = "1.0.0")]
pub struct Stderr {
    inner: Arc<ReentrantMutex<RefCell<Maybe<StderrRaw>>>>,
}

/// A locked reference to the `Stderr` handle.
///
/// This handle implements the `Write` trait and is constructed via
/// the [`Stderr::lock`] method.
///
/// [`Stderr::lock`]: struct.Stderr.html#method.lock
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
/// io::stderr().write(b"hello world")?;
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
/// handle.write(b"hello world")?;
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
    ///
    /// # Examples
    ///
    /// ```
    /// use std::io::{self, Write};
    ///
    /// fn foo() -> io::Result<()> {
    ///     let stderr = io::stderr();
    ///     let mut handle = stderr.lock();
    ///
    ///     handle.write(b"hello world")?;
    ///
    ///     Ok(())
    /// }
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn lock(&self) -> StderrLock {
        StderrLock { inner: self.inner.lock().unwrap_or_else(|e| e.into_inner()) }
    }
}

#[stable(feature = "std_debug", since = "1.15.0")]
impl fmt::Debug for Stderr {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.pad("Stderr { .. }")
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
        self.inner.borrow_mut().write(buf)
    }
    fn flush(&mut self) -> io::Result<()> {
        self.inner.borrow_mut().flush()
    }
}

#[stable(feature = "std_debug", since = "1.15.0")]
impl<'a> fmt::Debug for StderrLock<'a> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.pad("StderrLock { .. }")
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
pub fn set_panic(sink: Option<Box<Write + Send>>) -> Option<Box<Write + Send>> {
    use panicking::LOCAL_STDERR;
    use mem;
    LOCAL_STDERR.with(move |slot| {
        mem::replace(&mut *slot.borrow_mut(), sink)
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
pub fn set_print(sink: Option<Box<Write + Send>>) -> Option<Box<Write + Send>> {
    use mem;
    LOCAL_STDOUT.with(move |slot| {
        mem::replace(&mut *slot.borrow_mut(), sink)
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
    // As an implementation of the `println!` macro, we want to try our best to
    // not panic wherever possible and get the output somewhere. There are
    // currently two possible vectors for panics we take care of here:
    //
    // 1. If the TLS key for the local stdout has been destroyed, accessing it
    //    would cause a panic. Note that we just lump in the uninitialized case
    //    here for convenience, we're not trying to avoid a panic.
    // 2. If the local stdout is currently in use (e.g. we're in the middle of
    //    already printing) then accessing again would cause a panic.
    //
    // If, however, the actual I/O causes an error, we do indeed panic.
    let result = match LOCAL_STDOUT.state() {
        LocalKeyState::Uninitialized |
        LocalKeyState::Destroyed => stdout().write_fmt(args),
        LocalKeyState::Valid => {
            LOCAL_STDOUT.with(|s| {
                if let Ok(mut borrowed) = s.try_borrow_mut() {
                    if let Some(w) = borrowed.as_mut() {
                        return w.write_fmt(args);
                    }
                }
                stdout().write_fmt(args)
            })
        }
    };
    if let Err(e) = result {
        panic!("failed printing to stdout: {}", e);
    }
}

#[cfg(test)]
mod tests {
    use thread;
    use super::*;

    #[test]
    #[cfg_attr(target_os = "emscripten", ignore)]
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
