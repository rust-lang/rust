#![cfg_attr(test, allow(unused))]

#[cfg(test)]
mod tests;

use crate::cell::{Cell, RefCell};
use crate::fmt;
use crate::fs::File;
use crate::io::prelude::*;
use crate::io::{
    self, BorrowedCursor, BufReader, IoSlice, IoSliceMut, LineWriter, Lines, SpecReadByte,
};
use crate::panic::{RefUnwindSafe, UnwindSafe};
use crate::sync::atomic::{AtomicBool, Ordering};
use crate::sync::{Arc, Mutex, MutexGuard, OnceLock, ReentrantLock, ReentrantLockGuard};
use crate::sys::stdio;
use crate::thread::AccessError;

type LocalStream = Arc<Mutex<Vec<u8>>>;

thread_local! {
    /// Used by the test crate to capture the output of the print macros and panics.
    static OUTPUT_CAPTURE: Cell<Option<LocalStream>> = {
        Cell::new(None)
    }
}

/// Flag to indicate OUTPUT_CAPTURE is used.
///
/// If it is None and was never set on any thread, this flag is set to false,
/// and OUTPUT_CAPTURE can be safely ignored on all threads, saving some time
/// and memory registering an unused thread local.
///
/// Note about memory ordering: This contains information about whether a
/// thread local variable might be in use. Although this is a global flag, the
/// memory ordering between threads does not matter: we only want this flag to
/// have a consistent order between set_output_capture and print_to *within
/// the same thread*. Within the same thread, things always have a perfectly
/// consistent order. So Ordering::Relaxed is fine.
static OUTPUT_CAPTURE_USED: AtomicBool = AtomicBool::new(false);

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
#[unstable(feature = "libstd_sys_internals", issue = "none")]
const fn stdin_raw() -> StdinRaw {
    StdinRaw(stdio::Stdin::new())
}

/// Constructs a new raw handle to the standard output stream of this process.
///
/// The returned handle does not interact with any other handles created nor
/// handles returned by `std::io::stdout`. Note that data is buffered by the
/// `std::io::stdout` handles so writes which happen via this raw handle may
/// appear before previous writes.
///
/// The returned handle has no external synchronization or buffering layered on
/// top.
#[unstable(feature = "libstd_sys_internals", issue = "none")]
const fn stdout_raw() -> StdoutRaw {
    StdoutRaw(stdio::Stdout::new())
}

/// Constructs a new raw handle to the standard error stream of this process.
///
/// The returned handle does not interact with any other handles created nor
/// handles returned by `std::io::stderr`.
///
/// The returned handle has no external synchronization or buffering layered on
/// top.
#[unstable(feature = "libstd_sys_internals", issue = "none")]
const fn stderr_raw() -> StderrRaw {
    StderrRaw(stdio::Stderr::new())
}

impl Read for StdinRaw {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        handle_ebadf(self.0.read(buf), 0)
    }

    fn read_buf(&mut self, buf: BorrowedCursor<'_>) -> io::Result<()> {
        handle_ebadf(self.0.read_buf(buf), ())
    }

    fn read_vectored(&mut self, bufs: &mut [IoSliceMut<'_>]) -> io::Result<usize> {
        handle_ebadf(self.0.read_vectored(bufs), 0)
    }

    #[inline]
    fn is_read_vectored(&self) -> bool {
        self.0.is_read_vectored()
    }

    fn read_to_end(&mut self, buf: &mut Vec<u8>) -> io::Result<usize> {
        handle_ebadf(self.0.read_to_end(buf), 0)
    }

    fn read_to_string(&mut self, buf: &mut String) -> io::Result<usize> {
        handle_ebadf(self.0.read_to_string(buf), 0)
    }
}

impl Write for StdoutRaw {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        handle_ebadf(self.0.write(buf), buf.len())
    }

    fn write_vectored(&mut self, bufs: &[IoSlice<'_>]) -> io::Result<usize> {
        let total = || bufs.iter().map(|b| b.len()).sum();
        handle_ebadf_lazy(self.0.write_vectored(bufs), total)
    }

    #[inline]
    fn is_write_vectored(&self) -> bool {
        self.0.is_write_vectored()
    }

    fn flush(&mut self) -> io::Result<()> {
        handle_ebadf(self.0.flush(), ())
    }

    fn write_all(&mut self, buf: &[u8]) -> io::Result<()> {
        handle_ebadf(self.0.write_all(buf), ())
    }

    fn write_all_vectored(&mut self, bufs: &mut [IoSlice<'_>]) -> io::Result<()> {
        handle_ebadf(self.0.write_all_vectored(bufs), ())
    }

    fn write_fmt(&mut self, fmt: fmt::Arguments<'_>) -> io::Result<()> {
        handle_ebadf(self.0.write_fmt(fmt), ())
    }
}

impl Write for StderrRaw {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        handle_ebadf(self.0.write(buf), buf.len())
    }

    fn write_vectored(&mut self, bufs: &[IoSlice<'_>]) -> io::Result<usize> {
        let total = || bufs.iter().map(|b| b.len()).sum();
        handle_ebadf_lazy(self.0.write_vectored(bufs), total)
    }

    #[inline]
    fn is_write_vectored(&self) -> bool {
        self.0.is_write_vectored()
    }

    fn flush(&mut self) -> io::Result<()> {
        handle_ebadf(self.0.flush(), ())
    }

    fn write_all(&mut self, buf: &[u8]) -> io::Result<()> {
        handle_ebadf(self.0.write_all(buf), ())
    }

    fn write_all_vectored(&mut self, bufs: &mut [IoSlice<'_>]) -> io::Result<()> {
        handle_ebadf(self.0.write_all_vectored(bufs), ())
    }

    fn write_fmt(&mut self, fmt: fmt::Arguments<'_>) -> io::Result<()> {
        handle_ebadf(self.0.write_fmt(fmt), ())
    }
}

fn handle_ebadf<T>(r: io::Result<T>, default: T) -> io::Result<T> {
    match r {
        Err(ref e) if stdio::is_ebadf(e) => Ok(default),
        r => r,
    }
}

fn handle_ebadf_lazy<T>(r: io::Result<T>, default: impl FnOnce() -> T) -> io::Result<T> {
    match r {
        Err(ref e) if stdio::is_ebadf(e) => Ok(default()),
        r => r,
    }
}

/// A handle to the standard input stream of a process.
///
/// Each handle is a shared reference to a global buffer of input data to this
/// process. A handle can be `lock`'d to gain full access to [`BufRead`] methods
/// (e.g., `.lines()`). Reads to this handle are otherwise locked with respect
/// to other reads.
///
/// This handle implements the `Read` trait, but beware that concurrent reads
/// of `Stdin` must be executed with care.
///
/// Created by the [`io::stdin`] method.
///
/// [`io::stdin`]: stdin
///
/// ### Note: Windows Portability Considerations
///
/// When operating in a console, the Windows implementation of this stream does not support
/// non-UTF-8 byte sequences. Attempting to read bytes that are not valid UTF-8 will return
/// an error.
///
/// In a process with a detached console, such as one using
/// `#![windows_subsystem = "windows"]`, or in a child process spawned from such a process,
/// the contained handle will be null. In such cases, the standard library's `Read` and
/// `Write` will do nothing and silently succeed. All other I/O operations, via the
/// standard library or via raw Windows API calls, will fail.
///
/// # Examples
///
/// ```no_run
/// use std::io;
///
/// fn main() -> io::Result<()> {
///     let mut buffer = String::new();
///     let stdin = io::stdin(); // We get `Stdin` here.
///     stdin.read_line(&mut buffer)?;
///     Ok(())
/// }
/// ```
#[stable(feature = "rust1", since = "1.0.0")]
pub struct Stdin {
    inner: &'static Mutex<BufReader<StdinRaw>>,
}

/// A locked reference to the [`Stdin`] handle.
///
/// This handle implements both the [`Read`] and [`BufRead`] traits, and
/// is constructed via the [`Stdin::lock`] method.
///
/// ### Note: Windows Portability Considerations
///
/// When operating in a console, the Windows implementation of this stream does not support
/// non-UTF-8 byte sequences. Attempting to read bytes that are not valid UTF-8 will return
/// an error.
///
/// In a process with a detached console, such as one using
/// `#![windows_subsystem = "windows"]`, or in a child process spawned from such a process,
/// the contained handle will be null. In such cases, the standard library's `Read` and
/// `Write` will do nothing and silently succeed. All other I/O operations, via the
/// standard library or via raw Windows API calls, will fail.
///
/// # Examples
///
/// ```no_run
/// use std::io::{self, BufRead};
///
/// fn main() -> io::Result<()> {
///     let mut buffer = String::new();
///     let stdin = io::stdin(); // We get `Stdin` here.
///     {
///         let mut handle = stdin.lock(); // We get `StdinLock` here.
///         handle.read_line(&mut buffer)?;
///     } // `StdinLock` is dropped here.
///     Ok(())
/// }
/// ```
#[must_use = "if unused stdin will immediately unlock"]
#[stable(feature = "rust1", since = "1.0.0")]
pub struct StdinLock<'a> {
    inner: MutexGuard<'a, BufReader<StdinRaw>>,
}

/// Constructs a new handle to the standard input of the current process.
///
/// Each handle returned is a reference to a shared global buffer whose access
/// is synchronized via a mutex. If you need more explicit control over
/// locking, see the [`Stdin::lock`] method.
///
/// ### Note: Windows Portability Considerations
///
/// When operating in a console, the Windows implementation of this stream does not support
/// non-UTF-8 byte sequences. Attempting to read bytes that are not valid UTF-8 will return
/// an error.
///
/// In a process with a detached console, such as one using
/// `#![windows_subsystem = "windows"]`, or in a child process spawned from such a process,
/// the contained handle will be null. In such cases, the standard library's `Read` and
/// `Write` will do nothing and silently succeed. All other I/O operations, via the
/// standard library or via raw Windows API calls, will fail.
///
/// # Examples
///
/// Using implicit synchronization:
///
/// ```no_run
/// use std::io;
///
/// fn main() -> io::Result<()> {
///     let mut buffer = String::new();
///     io::stdin().read_line(&mut buffer)?;
///     Ok(())
/// }
/// ```
///
/// Using explicit synchronization:
///
/// ```no_run
/// use std::io::{self, BufRead};
///
/// fn main() -> io::Result<()> {
///     let mut buffer = String::new();
///     let stdin = io::stdin();
///     let mut handle = stdin.lock();
///
///     handle.read_line(&mut buffer)?;
///     Ok(())
/// }
/// ```
#[must_use]
#[stable(feature = "rust1", since = "1.0.0")]
pub fn stdin() -> Stdin {
    static INSTANCE: OnceLock<Mutex<BufReader<StdinRaw>>> = OnceLock::new();
    Stdin {
        inner: INSTANCE.get_or_init(|| {
            Mutex::new(BufReader::with_capacity(stdio::STDIN_BUF_SIZE, stdin_raw()))
        }),
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
    /// # Examples
    ///
    /// ```no_run
    /// use std::io::{self, BufRead};
    ///
    /// fn main() -> io::Result<()> {
    ///     let mut buffer = String::new();
    ///     let stdin = io::stdin();
    ///     let mut handle = stdin.lock();
    ///
    ///     handle.read_line(&mut buffer)?;
    ///     Ok(())
    /// }
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn lock(&self) -> StdinLock<'static> {
        // Locks this handle with 'static lifetime. This depends on the
        // implementation detail that the underlying `Mutex` is static.
        StdinLock { inner: self.inner.lock().unwrap_or_else(|e| e.into_inner()) }
    }

    /// Locks this handle and reads a line of input, appending it to the specified buffer.
    ///
    /// For detailed semantics of this method, see the documentation on
    /// [`BufRead::read_line`]. In particular:
    /// * Previous content of the buffer will be preserved. To avoid appending
    ///   to the buffer, you need to [`clear`] it first.
    /// * The trailing newline character, if any, is included in the buffer.
    ///
    /// [`clear`]: String::clear
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use std::io;
    ///
    /// let mut input = String::new();
    /// match io::stdin().read_line(&mut input) {
    ///     Ok(n) => {
    ///         println!("{n} bytes read");
    ///         println!("{input}");
    ///     }
    ///     Err(error) => println!("error: {error}"),
    /// }
    /// ```
    ///
    /// You can run the example one of two ways:
    ///
    /// - Pipe some text to it, e.g., `printf foo | path/to/executable`
    /// - Give it text interactively by running the executable directly,
    ///   in which case it will wait for the Enter key to be pressed before
    ///   continuing
    #[stable(feature = "rust1", since = "1.0.0")]
    #[rustc_confusables("get_line")]
    pub fn read_line(&self, buf: &mut String) -> io::Result<usize> {
        self.lock().read_line(buf)
    }

    /// Consumes this handle and returns an iterator over input lines.
    ///
    /// For detailed semantics of this method, see the documentation on
    /// [`BufRead::lines`].
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use std::io;
    ///
    /// let lines = io::stdin().lines();
    /// for line in lines {
    ///     println!("got a line: {}", line.unwrap());
    /// }
    /// ```
    #[must_use = "`self` will be dropped if the result is not used"]
    #[stable(feature = "stdin_forwarders", since = "1.62.0")]
    pub fn lines(self) -> Lines<StdinLock<'static>> {
        self.lock().lines()
    }
}

#[stable(feature = "std_debug", since = "1.16.0")]
impl fmt::Debug for Stdin {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Stdin").finish_non_exhaustive()
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl Read for Stdin {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        self.lock().read(buf)
    }
    fn read_buf(&mut self, buf: BorrowedCursor<'_>) -> io::Result<()> {
        self.lock().read_buf(buf)
    }
    fn read_vectored(&mut self, bufs: &mut [IoSliceMut<'_>]) -> io::Result<usize> {
        self.lock().read_vectored(bufs)
    }
    #[inline]
    fn is_read_vectored(&self) -> bool {
        self.lock().is_read_vectored()
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
    fn read_buf_exact(&mut self, cursor: BorrowedCursor<'_>) -> io::Result<()> {
        self.lock().read_buf_exact(cursor)
    }
}

#[stable(feature = "read_shared_stdin", since = "1.78.0")]
impl Read for &Stdin {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        self.lock().read(buf)
    }
    fn read_buf(&mut self, buf: BorrowedCursor<'_>) -> io::Result<()> {
        self.lock().read_buf(buf)
    }
    fn read_vectored(&mut self, bufs: &mut [IoSliceMut<'_>]) -> io::Result<usize> {
        self.lock().read_vectored(bufs)
    }
    #[inline]
    fn is_read_vectored(&self) -> bool {
        self.lock().is_read_vectored()
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
    fn read_buf_exact(&mut self, cursor: BorrowedCursor<'_>) -> io::Result<()> {
        self.lock().read_buf_exact(cursor)
    }
}

// only used by platform-dependent io::copy specializations, i.e. unused on some platforms
#[cfg(any(target_os = "linux", target_os = "android"))]
impl StdinLock<'_> {
    pub(crate) fn as_mut_buf(&mut self) -> &mut BufReader<impl Read> {
        &mut self.inner
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl Read for StdinLock<'_> {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        self.inner.read(buf)
    }

    fn read_buf(&mut self, buf: BorrowedCursor<'_>) -> io::Result<()> {
        self.inner.read_buf(buf)
    }

    fn read_vectored(&mut self, bufs: &mut [IoSliceMut<'_>]) -> io::Result<usize> {
        self.inner.read_vectored(bufs)
    }

    #[inline]
    fn is_read_vectored(&self) -> bool {
        self.inner.is_read_vectored()
    }

    fn read_to_end(&mut self, buf: &mut Vec<u8>) -> io::Result<usize> {
        self.inner.read_to_end(buf)
    }

    fn read_to_string(&mut self, buf: &mut String) -> io::Result<usize> {
        self.inner.read_to_string(buf)
    }

    fn read_exact(&mut self, buf: &mut [u8]) -> io::Result<()> {
        self.inner.read_exact(buf)
    }

    fn read_buf_exact(&mut self, cursor: BorrowedCursor<'_>) -> io::Result<()> {
        self.inner.read_buf_exact(cursor)
    }
}

impl SpecReadByte for StdinLock<'_> {
    #[inline]
    fn spec_read_byte(&mut self) -> Option<io::Result<u8>> {
        BufReader::spec_read_byte(&mut *self.inner)
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl BufRead for StdinLock<'_> {
    fn fill_buf(&mut self) -> io::Result<&[u8]> {
        self.inner.fill_buf()
    }

    fn consume(&mut self, n: usize) {
        self.inner.consume(n)
    }

    fn read_until(&mut self, byte: u8, buf: &mut Vec<u8>) -> io::Result<usize> {
        self.inner.read_until(byte, buf)
    }

    fn read_line(&mut self, buf: &mut String) -> io::Result<usize> {
        self.inner.read_line(buf)
    }
}

#[stable(feature = "std_debug", since = "1.16.0")]
impl fmt::Debug for StdinLock<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("StdinLock").finish_non_exhaustive()
    }
}

/// A handle to the global standard output stream of the current process.
///
/// Each handle shares a global buffer of data to be written to the standard
/// output stream. Access is also synchronized via a lock and explicit control
/// over locking is available via the [`lock`] method.
///
/// Created by the [`io::stdout`] method.
///
/// ### Note: Windows Portability Considerations
///
/// When operating in a console, the Windows implementation of this stream does not support
/// non-UTF-8 byte sequences. Attempting to write bytes that are not valid UTF-8 will return
/// an error.
///
/// In a process with a detached console, such as one using
/// `#![windows_subsystem = "windows"]`, or in a child process spawned from such a process,
/// the contained handle will be null. In such cases, the standard library's `Read` and
/// `Write` will do nothing and silently succeed. All other I/O operations, via the
/// standard library or via raw Windows API calls, will fail.
///
/// [`lock`]: Stdout::lock
/// [`io::stdout`]: stdout
#[stable(feature = "rust1", since = "1.0.0")]
pub struct Stdout {
    // FIXME: this should be LineWriter or BufWriter depending on the state of
    //        stdout (tty or not). Note that if this is not line buffered it
    //        should also flush-on-panic or some form of flush-on-abort.
    inner: &'static ReentrantLock<RefCell<LineWriter<StdoutRaw>>>,
}

/// A locked reference to the [`Stdout`] handle.
///
/// This handle implements the [`Write`] trait, and is constructed via
/// the [`Stdout::lock`] method. See its documentation for more.
///
/// ### Note: Windows Portability Considerations
///
/// When operating in a console, the Windows implementation of this stream does not support
/// non-UTF-8 byte sequences. Attempting to write bytes that are not valid UTF-8 will return
/// an error.
///
/// In a process with a detached console, such as one using
/// `#![windows_subsystem = "windows"]`, or in a child process spawned from such a process,
/// the contained handle will be null. In such cases, the standard library's `Read` and
/// `Write` will do nothing and silently succeed. All other I/O operations, via the
/// standard library or via raw Windows API calls, will fail.
#[must_use = "if unused stdout will immediately unlock"]
#[stable(feature = "rust1", since = "1.0.0")]
pub struct StdoutLock<'a> {
    inner: ReentrantLockGuard<'a, RefCell<LineWriter<StdoutRaw>>>,
}

static STDOUT: OnceLock<ReentrantLock<RefCell<LineWriter<StdoutRaw>>>> = OnceLock::new();

/// Constructs a new handle to the standard output of the current process.
///
/// Each handle returned is a reference to a shared global buffer whose access
/// is synchronized via a mutex. If you need more explicit control over
/// locking, see the [`Stdout::lock`] method.
///
/// ### Note: Windows Portability Considerations
///
/// When operating in a console, the Windows implementation of this stream does not support
/// non-UTF-8 byte sequences. Attempting to write bytes that are not valid UTF-8 will return
/// an error.
///
/// In a process with a detached console, such as one using
/// `#![windows_subsystem = "windows"]`, or in a child process spawned from such a process,
/// the contained handle will be null. In such cases, the standard library's `Read` and
/// `Write` will do nothing and silently succeed. All other I/O operations, via the
/// standard library or via raw Windows API calls, will fail.
///
/// # Examples
///
/// Using implicit synchronization:
///
/// ```no_run
/// use std::io::{self, Write};
///
/// fn main() -> io::Result<()> {
///     io::stdout().write_all(b"hello world")?;
///
///     Ok(())
/// }
/// ```
///
/// Using explicit synchronization:
///
/// ```no_run
/// use std::io::{self, Write};
///
/// fn main() -> io::Result<()> {
///     let stdout = io::stdout();
///     let mut handle = stdout.lock();
///
///     handle.write_all(b"hello world")?;
///
///     Ok(())
/// }
/// ```
#[must_use]
#[stable(feature = "rust1", since = "1.0.0")]
#[cfg_attr(not(test), rustc_diagnostic_item = "io_stdout")]
pub fn stdout() -> Stdout {
    Stdout {
        inner: STDOUT
            .get_or_init(|| ReentrantLock::new(RefCell::new(LineWriter::new(stdout_raw())))),
    }
}

// Flush the data and disable buffering during shutdown
// by replacing the line writer by one with zero
// buffering capacity.
pub fn cleanup() {
    let mut initialized = false;
    let stdout = STDOUT.get_or_init(|| {
        initialized = true;
        ReentrantLock::new(RefCell::new(LineWriter::with_capacity(0, stdout_raw())))
    });

    if !initialized {
        // The buffer was previously initialized, overwrite it here.
        // We use try_lock() instead of lock(), because someone
        // might have leaked a StdoutLock, which would
        // otherwise cause a deadlock here.
        if let Some(lock) = stdout.try_lock() {
            *lock.borrow_mut() = LineWriter::with_capacity(0, stdout_raw());
        }
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
    /// ```no_run
    /// use std::io::{self, Write};
    ///
    /// fn main() -> io::Result<()> {
    ///     let mut stdout = io::stdout().lock();
    ///
    ///     stdout.write_all(b"hello world")?;
    ///
    ///     Ok(())
    /// }
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn lock(&self) -> StdoutLock<'static> {
        // Locks this handle with 'static lifetime. This depends on the
        // implementation detail that the underlying `ReentrantMutex` is
        // static.
        StdoutLock { inner: self.inner.lock() }
    }
}

#[stable(feature = "catch_unwind", since = "1.9.0")]
impl UnwindSafe for Stdout {}

#[stable(feature = "catch_unwind", since = "1.9.0")]
impl RefUnwindSafe for Stdout {}

#[stable(feature = "std_debug", since = "1.16.0")]
impl fmt::Debug for Stdout {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Stdout").finish_non_exhaustive()
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl Write for Stdout {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        (&*self).write(buf)
    }
    fn write_vectored(&mut self, bufs: &[IoSlice<'_>]) -> io::Result<usize> {
        (&*self).write_vectored(bufs)
    }
    #[inline]
    fn is_write_vectored(&self) -> bool {
        io::Write::is_write_vectored(&&*self)
    }
    fn flush(&mut self) -> io::Result<()> {
        (&*self).flush()
    }
    fn write_all(&mut self, buf: &[u8]) -> io::Result<()> {
        (&*self).write_all(buf)
    }
    fn write_all_vectored(&mut self, bufs: &mut [IoSlice<'_>]) -> io::Result<()> {
        (&*self).write_all_vectored(bufs)
    }
    fn write_fmt(&mut self, args: fmt::Arguments<'_>) -> io::Result<()> {
        (&*self).write_fmt(args)
    }
}

#[stable(feature = "write_mt", since = "1.48.0")]
impl Write for &Stdout {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        self.lock().write(buf)
    }
    fn write_vectored(&mut self, bufs: &[IoSlice<'_>]) -> io::Result<usize> {
        self.lock().write_vectored(bufs)
    }
    #[inline]
    fn is_write_vectored(&self) -> bool {
        self.lock().is_write_vectored()
    }
    fn flush(&mut self) -> io::Result<()> {
        self.lock().flush()
    }
    fn write_all(&mut self, buf: &[u8]) -> io::Result<()> {
        self.lock().write_all(buf)
    }
    fn write_all_vectored(&mut self, bufs: &mut [IoSlice<'_>]) -> io::Result<()> {
        self.lock().write_all_vectored(bufs)
    }
    fn write_fmt(&mut self, args: fmt::Arguments<'_>) -> io::Result<()> {
        self.lock().write_fmt(args)
    }
}

#[stable(feature = "catch_unwind", since = "1.9.0")]
impl UnwindSafe for StdoutLock<'_> {}

#[stable(feature = "catch_unwind", since = "1.9.0")]
impl RefUnwindSafe for StdoutLock<'_> {}

#[stable(feature = "rust1", since = "1.0.0")]
impl Write for StdoutLock<'_> {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        self.inner.borrow_mut().write(buf)
    }
    fn write_vectored(&mut self, bufs: &[IoSlice<'_>]) -> io::Result<usize> {
        self.inner.borrow_mut().write_vectored(bufs)
    }
    #[inline]
    fn is_write_vectored(&self) -> bool {
        self.inner.borrow_mut().is_write_vectored()
    }
    fn flush(&mut self) -> io::Result<()> {
        self.inner.borrow_mut().flush()
    }
    fn write_all(&mut self, buf: &[u8]) -> io::Result<()> {
        self.inner.borrow_mut().write_all(buf)
    }
    fn write_all_vectored(&mut self, bufs: &mut [IoSlice<'_>]) -> io::Result<()> {
        self.inner.borrow_mut().write_all_vectored(bufs)
    }
}

#[stable(feature = "std_debug", since = "1.16.0")]
impl fmt::Debug for StdoutLock<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("StdoutLock").finish_non_exhaustive()
    }
}

/// A handle to the standard error stream of a process.
///
/// For more information, see the [`io::stderr`] method.
///
/// [`io::stderr`]: stderr
///
/// ### Note: Windows Portability Considerations
///
/// When operating in a console, the Windows implementation of this stream does not support
/// non-UTF-8 byte sequences. Attempting to write bytes that are not valid UTF-8 will return
/// an error.
///
/// In a process with a detached console, such as one using
/// `#![windows_subsystem = "windows"]`, or in a child process spawned from such a process,
/// the contained handle will be null. In such cases, the standard library's `Read` and
/// `Write` will do nothing and silently succeed. All other I/O operations, via the
/// standard library or via raw Windows API calls, will fail.
#[stable(feature = "rust1", since = "1.0.0")]
pub struct Stderr {
    inner: &'static ReentrantLock<RefCell<StderrRaw>>,
}

/// A locked reference to the [`Stderr`] handle.
///
/// This handle implements the [`Write`] trait and is constructed via
/// the [`Stderr::lock`] method. See its documentation for more.
///
/// ### Note: Windows Portability Considerations
///
/// When operating in a console, the Windows implementation of this stream does not support
/// non-UTF-8 byte sequences. Attempting to write bytes that are not valid UTF-8 will return
/// an error.
///
/// In a process with a detached console, such as one using
/// `#![windows_subsystem = "windows"]`, or in a child process spawned from such a process,
/// the contained handle will be null. In such cases, the standard library's `Read` and
/// `Write` will do nothing and silently succeed. All other I/O operations, via the
/// standard library or via raw Windows API calls, will fail.
#[must_use = "if unused stderr will immediately unlock"]
#[stable(feature = "rust1", since = "1.0.0")]
pub struct StderrLock<'a> {
    inner: ReentrantLockGuard<'a, RefCell<StderrRaw>>,
}

/// Constructs a new handle to the standard error of the current process.
///
/// This handle is not buffered.
///
/// ### Note: Windows Portability Considerations
///
/// When operating in a console, the Windows implementation of this stream does not support
/// non-UTF-8 byte sequences. Attempting to write bytes that are not valid UTF-8 will return
/// an error.
///
/// In a process with a detached console, such as one using
/// `#![windows_subsystem = "windows"]`, or in a child process spawned from such a process,
/// the contained handle will be null. In such cases, the standard library's `Read` and
/// `Write` will do nothing and silently succeed. All other I/O operations, via the
/// standard library or via raw Windows API calls, will fail.
///
/// # Examples
///
/// Using implicit synchronization:
///
/// ```no_run
/// use std::io::{self, Write};
///
/// fn main() -> io::Result<()> {
///     io::stderr().write_all(b"hello world")?;
///
///     Ok(())
/// }
/// ```
///
/// Using explicit synchronization:
///
/// ```no_run
/// use std::io::{self, Write};
///
/// fn main() -> io::Result<()> {
///     let stderr = io::stderr();
///     let mut handle = stderr.lock();
///
///     handle.write_all(b"hello world")?;
///
///     Ok(())
/// }
/// ```
#[must_use]
#[stable(feature = "rust1", since = "1.0.0")]
#[cfg_attr(not(test), rustc_diagnostic_item = "io_stderr")]
pub fn stderr() -> Stderr {
    // Note that unlike `stdout()` we don't use `at_exit` here to register a
    // destructor. Stderr is not buffered, so there's no need to run a
    // destructor for flushing the buffer
    static INSTANCE: ReentrantLock<RefCell<StderrRaw>> =
        ReentrantLock::new(RefCell::new(stderr_raw()));

    Stderr { inner: &INSTANCE }
}

impl Stderr {
    /// Locks this handle to the standard error stream, returning a writable
    /// guard.
    ///
    /// The lock is released when the returned lock goes out of scope. The
    /// returned guard also implements the [`Write`] trait for writing data.
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
    ///     handle.write_all(b"hello world")?;
    ///
    ///     Ok(())
    /// }
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn lock(&self) -> StderrLock<'static> {
        // Locks this handle with 'static lifetime. This depends on the
        // implementation detail that the underlying `ReentrantMutex` is
        // static.
        StderrLock { inner: self.inner.lock() }
    }
}

#[stable(feature = "catch_unwind", since = "1.9.0")]
impl UnwindSafe for Stderr {}

#[stable(feature = "catch_unwind", since = "1.9.0")]
impl RefUnwindSafe for Stderr {}

#[stable(feature = "std_debug", since = "1.16.0")]
impl fmt::Debug for Stderr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Stderr").finish_non_exhaustive()
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl Write for Stderr {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        (&*self).write(buf)
    }
    fn write_vectored(&mut self, bufs: &[IoSlice<'_>]) -> io::Result<usize> {
        (&*self).write_vectored(bufs)
    }
    #[inline]
    fn is_write_vectored(&self) -> bool {
        io::Write::is_write_vectored(&&*self)
    }
    fn flush(&mut self) -> io::Result<()> {
        (&*self).flush()
    }
    fn write_all(&mut self, buf: &[u8]) -> io::Result<()> {
        (&*self).write_all(buf)
    }
    fn write_all_vectored(&mut self, bufs: &mut [IoSlice<'_>]) -> io::Result<()> {
        (&*self).write_all_vectored(bufs)
    }
    fn write_fmt(&mut self, args: fmt::Arguments<'_>) -> io::Result<()> {
        (&*self).write_fmt(args)
    }
}

#[stable(feature = "write_mt", since = "1.48.0")]
impl Write for &Stderr {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        self.lock().write(buf)
    }
    fn write_vectored(&mut self, bufs: &[IoSlice<'_>]) -> io::Result<usize> {
        self.lock().write_vectored(bufs)
    }
    #[inline]
    fn is_write_vectored(&self) -> bool {
        self.lock().is_write_vectored()
    }
    fn flush(&mut self) -> io::Result<()> {
        self.lock().flush()
    }
    fn write_all(&mut self, buf: &[u8]) -> io::Result<()> {
        self.lock().write_all(buf)
    }
    fn write_all_vectored(&mut self, bufs: &mut [IoSlice<'_>]) -> io::Result<()> {
        self.lock().write_all_vectored(bufs)
    }
    fn write_fmt(&mut self, args: fmt::Arguments<'_>) -> io::Result<()> {
        self.lock().write_fmt(args)
    }
}

#[stable(feature = "catch_unwind", since = "1.9.0")]
impl UnwindSafe for StderrLock<'_> {}

#[stable(feature = "catch_unwind", since = "1.9.0")]
impl RefUnwindSafe for StderrLock<'_> {}

#[stable(feature = "rust1", since = "1.0.0")]
impl Write for StderrLock<'_> {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        self.inner.borrow_mut().write(buf)
    }
    fn write_vectored(&mut self, bufs: &[IoSlice<'_>]) -> io::Result<usize> {
        self.inner.borrow_mut().write_vectored(bufs)
    }
    #[inline]
    fn is_write_vectored(&self) -> bool {
        self.inner.borrow_mut().is_write_vectored()
    }
    fn flush(&mut self) -> io::Result<()> {
        self.inner.borrow_mut().flush()
    }
    fn write_all(&mut self, buf: &[u8]) -> io::Result<()> {
        self.inner.borrow_mut().write_all(buf)
    }
    fn write_all_vectored(&mut self, bufs: &mut [IoSlice<'_>]) -> io::Result<()> {
        self.inner.borrow_mut().write_all_vectored(bufs)
    }
}

#[stable(feature = "std_debug", since = "1.16.0")]
impl fmt::Debug for StderrLock<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("StderrLock").finish_non_exhaustive()
    }
}

/// Sets the thread-local output capture buffer and returns the old one.
#[unstable(
    feature = "internal_output_capture",
    reason = "this function is meant for use in the test crate \
        and may disappear in the future",
    issue = "none"
)]
#[doc(hidden)]
pub fn set_output_capture(sink: Option<LocalStream>) -> Option<LocalStream> {
    try_set_output_capture(sink).expect(
        "cannot access a Thread Local Storage value \
         during or after destruction",
    )
}

/// Tries to set the thread-local output capture buffer and returns the old one.
/// This may fail once thread-local destructors are called. It's used in panic
/// handling instead of `set_output_capture`.
#[unstable(
    feature = "internal_output_capture",
    reason = "this function is meant for use in the test crate \
    and may disappear in the future",
    issue = "none"
)]
#[doc(hidden)]
pub fn try_set_output_capture(
    sink: Option<LocalStream>,
) -> Result<Option<LocalStream>, AccessError> {
    if sink.is_none() && !OUTPUT_CAPTURE_USED.load(Ordering::Relaxed) {
        // OUTPUT_CAPTURE is definitely None since OUTPUT_CAPTURE_USED is false.
        return Ok(None);
    }
    OUTPUT_CAPTURE_USED.store(true, Ordering::Relaxed);
    OUTPUT_CAPTURE.try_with(move |slot| slot.replace(sink))
}

/// Writes `args` to the capture buffer if enabled and possible, or `global_s`
/// otherwise. `label` identifies the stream in a panic message.
///
/// This function is used to print error messages, so it takes extra
/// care to avoid causing a panic when `OUTPUT_CAPTURE` is unusable.
/// For instance, if the TLS key for output capturing is already destroyed, or
/// if the local stream is in use by another thread, it will just fall back to
/// the global stream.
///
/// However, if the actual I/O causes an error, this function does panic.
///
/// Writing to non-blocking stdout/stderr can cause an error, which will lead
/// this function to panic.
fn print_to<T>(args: fmt::Arguments<'_>, global_s: fn() -> T, label: &str)
where
    T: Write,
{
    if print_to_buffer_if_capture_used(args) {
        // Successfully wrote to capture buffer.
        return;
    }

    if let Err(e) = global_s().write_fmt(args) {
        panic!("failed printing to {label}: {e}");
    }
}

fn print_to_buffer_if_capture_used(args: fmt::Arguments<'_>) -> bool {
    OUTPUT_CAPTURE_USED.load(Ordering::Relaxed)
        && OUTPUT_CAPTURE.try_with(|s| {
            // Note that we completely remove a local sink to write to in case
            // our printing recursively panics/prints, so the recursive
            // panic/print goes to the global sink instead of our local sink.
            s.take().map(|w| {
                let _ = w.lock().unwrap_or_else(|e| e.into_inner()).write_fmt(args);
                s.set(Some(w));
            })
        }) == Ok(Some(()))
}

/// Used by impl Termination for Result to print error after `main` or a test
/// has returned. Should avoid panicking, although we can't help it if one of
/// the Display impls inside args decides to.
pub(crate) fn attempt_print_to_stderr(args: fmt::Arguments<'_>) {
    if print_to_buffer_if_capture_used(args) {
        return;
    }

    // Ignore error if the write fails, for example because stderr is already
    // closed. There is not much point panicking at this point.
    let _ = stderr().write_fmt(args);
}

/// Trait to determine if a descriptor/handle refers to a terminal/tty.
#[stable(feature = "is_terminal", since = "1.70.0")]
pub trait IsTerminal: crate::sealed::Sealed {
    /// Returns `true` if the descriptor/handle refers to a terminal/tty.
    ///
    /// On platforms where Rust does not know how to detect a terminal yet, this will return
    /// `false`. This will also return `false` if an unexpected error occurred, such as from
    /// passing an invalid file descriptor.
    ///
    /// # Platform-specific behavior
    ///
    /// On Windows, in addition to detecting consoles, this currently uses some heuristics to
    /// detect older msys/cygwin/mingw pseudo-terminals based on device name: devices with names
    /// starting with `msys-` or `cygwin-` and ending in `-pty` will be considered terminals.
    /// Note that this [may change in the future][changes].
    ///
    /// # Examples
    ///
    /// An example of a type for which `IsTerminal` is implemented is [`Stdin`]:
    ///
    /// ```no_run
    /// use std::io::{self, IsTerminal, Write};
    ///
    /// fn main() -> io::Result<()> {
    ///     let stdin = io::stdin();
    ///
    ///     // Indicate that the user is prompted for input, if this is a terminal.
    ///     if stdin.is_terminal() {
    ///         print!("> ");
    ///         io::stdout().flush()?;
    ///     }
    ///
    ///     let mut name = String::new();
    ///     let _ = stdin.read_line(&mut name)?;
    ///
    ///     println!("Hello {}", name.trim_end());
    ///
    ///     Ok(())
    /// }
    /// ```
    ///
    /// The example can be run in two ways:
    ///
    /// - If you run this example by piping some text to it, e.g. `echo "foo" | path/to/executable`
    ///   it will print: `Hello foo`.
    /// - If you instead run the example interactively by running `path/to/executable` directly, it will
    ///   prompt for input.
    ///
    /// [changes]: io#platform-specific-behavior
    /// [`Stdin`]: crate::io::Stdin
    #[doc(alias = "isatty")]
    #[stable(feature = "is_terminal", since = "1.70.0")]
    fn is_terminal(&self) -> bool;
}

macro_rules! impl_is_terminal {
    ($($t:ty),*$(,)?) => {$(
        #[unstable(feature = "sealed", issue = "none")]
        impl crate::sealed::Sealed for $t {}

        #[stable(feature = "is_terminal", since = "1.70.0")]
        impl IsTerminal for $t {
            #[inline]
            fn is_terminal(&self) -> bool {
                crate::sys::io::is_terminal(self)
            }
        }
    )*}
}

impl_is_terminal!(File, Stdin, StdinLock<'_>, Stdout, StdoutLock<'_>, Stderr, StderrLock<'_>);

#[unstable(
    feature = "print_internals",
    reason = "implementation detail which may disappear or be replaced at any time",
    issue = "none"
)]
#[doc(hidden)]
#[cfg(not(test))]
pub fn _print(args: fmt::Arguments<'_>) {
    print_to(args, stdout, "stdout");
}

#[unstable(
    feature = "print_internals",
    reason = "implementation detail which may disappear or be replaced at any time",
    issue = "none"
)]
#[doc(hidden)]
#[cfg(not(test))]
pub fn _eprint(args: fmt::Arguments<'_>) {
    print_to(args, stderr, "stderr");
}

#[cfg(test)]
pub use realstd::io::{_eprint, _print};
