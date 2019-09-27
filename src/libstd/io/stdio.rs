#![cfg_attr(test, allow(unused))]

use crate::io::prelude::*;

use crate::cell::RefCell;
use crate::fmt;
use crate::io::lazy::Lazy;
use crate::io::{self, Initializer, BufReader, BufWriter, LineWriter, IoSlice, IoSliceMut};
use crate::sync::{Arc, Mutex, MutexGuard};
use crate::sys::stdio;
use crate::sys_common::remutex::{ReentrantMutex, ReentrantMutexGuard};
use crate::thread::LocalKey;

thread_local! {
    /// Stdout used by print! and println! macros
    static LOCAL_STDOUT: RefCell<Option<Box<dyn Write + Send>>> = {
        RefCell::new(None)
    }
}

thread_local! {
    /// Stderr used by eprint! and eprintln! macros, and panics
    static LOCAL_STDERR: RefCell<Option<Box<dyn Write + Send>>> = {
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

    fn read_vectored(&mut self, bufs: &mut [IoSliceMut<'_>]) -> io::Result<usize> {
        self.0.read_vectored(bufs)
    }

    #[inline]
    unsafe fn initializer(&self) -> Initializer {
        Initializer::nop()
    }
}
impl StdoutRaw {
    fn should_be_line_buffered(&self) -> bool {
        self.0.should_be_line_buffered()
    }
}
impl Write for StdoutRaw {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> { self.0.write(buf) }

    fn write_vectored(&mut self, bufs: &[IoSlice<'_>]) -> io::Result<usize> {
        self.0.write_vectored(bufs)
    }

    fn flush(&mut self) -> io::Result<()> { self.0.flush() }
}
impl Write for StderrRaw {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> { self.0.write(buf) }

    fn write_vectored(&mut self, bufs: &[IoSlice<'_>]) -> io::Result<usize> {
        self.0.write_vectored(bufs)
    }

    fn flush(&mut self) -> io::Result<()> { self.0.flush() }
}

/// Maps `EBADF` errors for reading and writing to success.
struct HandleEbadf<T>(T);

impl<W: Write> Write for HandleEbadf<W> {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        handle_ebadf(self.0.write(buf), buf.len())
    }
    fn write_vectored(&mut self, bufs: &[IoSlice<'_>]) -> io::Result<usize> {
        let total = bufs.iter().map(|b| b.len()).sum();
        handle_ebadf(self.0.write_vectored(bufs), total)
    }
    fn flush(&mut self) -> io::Result<()> {
        handle_ebadf(self.0.flush(), ())
    }
}

impl<R: Read> Read for HandleEbadf<R> {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        handle_ebadf(self.0.read(buf), 0)
    }
    fn read_vectored(&mut self, bufs: &mut [IoSliceMut<'_>]) -> io::Result<usize> {
        handle_ebadf(self.0.read_vectored(bufs), 0)
    }
    unsafe fn initializer(&self) -> Initializer {
        self.0.initializer()
    }
}

fn handle_ebadf<T>(r: io::Result<T>, default: T) -> io::Result<T> {
    match r {
        Err(ref e) if stdio::is_ebadf(e) => Ok(default),
        r => r
    }
}

enum StdioBufferKind {
    LineBuffered,
    Buffered,
    Unbuffered,
}

/// A reader that can either be buffered or fake.
///
/// If it is fake, all reads succeed and indicate end-of-file.
enum StdioReader<R: Read> {
    Buffered(BufReader<HandleEbadf<R>>),
    Fake,
}

impl<R: Read> StdioReader<R> {
    fn new_buffered(reader: R) -> StdioReader<R> {
        let cap = stdio::STDIN_BUF_SIZE;
        StdioReader::Buffered(BufReader::with_capacity(cap, HandleEbadf(reader)))
    }
    fn new_fake() -> StdioReader<R> {
        StdioReader::Fake
    }
}

impl<R: Read> Read for StdioReader<R> {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        match self {
            StdioReader::Buffered(i) => i.read(buf),
            StdioReader::Fake => Ok(0),
        }
    }
    fn read_vectored(&mut self, bufs: &mut [io::IoSliceMut<'_>]) -> io::Result<usize> {
        match self {
            StdioReader::Buffered(i) => i.read_vectored(bufs),
            StdioReader::Fake => Ok(0),
        }
    }
    #[inline]
    unsafe fn initializer(&self) -> Initializer {
        match self {
            StdioReader::Buffered(i) => i.initializer(),
            StdioReader::Fake => Initializer::nop(),
        }
    }
}

impl<R: Read> BufRead for StdioReader<R> {
    fn fill_buf(&mut self) -> io::Result<&[u8]> {
        match self {
            StdioReader::Buffered(i) => i.fill_buf(),
            StdioReader::Fake => Ok(&[]),
        }
    }
    fn consume(&mut self, amt: usize) {
        match self {
            StdioReader::Buffered(i) => i.consume(amt),
            StdioReader::Fake => (),
        }
    }
}

/// A writer that can either be line-buffered, buffered, unbuffered or fake.
///
/// If it is fake, all outputs just succeed and are dropped silently.
enum StdioWriter<W: Write> {
    LineBuffered(LineWriter<HandleEbadf<W>>),
    Buffered(BufWriter<HandleEbadf<W>>),
    Unbuffered(HandleEbadf<W>),
    Fake,
}

impl<W: Write> StdioWriter<W> {
    fn new(writer: W, kind: StdioBufferKind) -> StdioWriter<W> {
        let writer = HandleEbadf(writer);
        match kind {
            StdioBufferKind::LineBuffered =>
                StdioWriter::LineBuffered(LineWriter::new(writer)),
            StdioBufferKind::Buffered =>
                StdioWriter::Buffered(BufWriter::new(writer)),
            StdioBufferKind::Unbuffered =>
                StdioWriter::Unbuffered(writer),
        }
    }
    fn new_fake() -> StdioWriter<W> {
        StdioWriter::Fake
    }
}

impl<W: Write> Write for StdioWriter<W> {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        match self {
            StdioWriter::LineBuffered(i) => i.write(buf),
            StdioWriter::Buffered(i) => i.write(buf),
            StdioWriter::Unbuffered(i) => i.write(buf),
            StdioWriter::Fake => Ok(buf.len()),
        }
    }
    fn write_vectored(&mut self, bufs: &[io::IoSlice<'_>]) -> io::Result<usize> {
        match self {
            StdioWriter::LineBuffered(i) => i.write_vectored(bufs),
            StdioWriter::Buffered(i) => i.write_vectored(bufs),
            StdioWriter::Unbuffered(i) => i.write_vectored(bufs),
            StdioWriter::Fake => Ok(bufs.iter().map(|b| b.len()).sum()),
        }
    }
    fn flush(&mut self) -> io::Result<()> {
        match self {
            StdioWriter::LineBuffered(i) => i.flush(),
            StdioWriter::Buffered(i) => i.flush(),
            StdioWriter::Unbuffered(i) => i.flush(),
            StdioWriter::Fake => Ok(()),
        }
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
/// [`io::stdin`]: fn.stdin.html
/// [`BufRead`]: trait.BufRead.html
///
/// ### Note: Windows Portability Consideration
/// When operating in a console, the Windows implementation of this stream does not support
/// non-UTF-8 byte sequences. Attempting to read bytes that are not valid UTF-8 will return
/// an error.
#[stable(feature = "rust1", since = "1.0.0")]
pub struct Stdin {
    inner: Arc<Mutex<StdioReader<StdinRaw>>>,
}

/// A locked reference to the `Stdin` handle.
///
/// This handle implements both the [`Read`] and [`BufRead`] traits, and
/// is constructed via the [`Stdin::lock`] method.
///
/// [`Read`]: trait.Read.html
/// [`BufRead`]: trait.BufRead.html
/// [`Stdin::lock`]: struct.Stdin.html#method.lock
///
/// ### Note: Windows Portability Consideration
/// When operating in a console, the Windows implementation of this stream does not support
/// non-UTF-8 byte sequences. Attempting to read bytes that are not valid UTF-8 will return
/// an error.
#[stable(feature = "rust1", since = "1.0.0")]
pub struct StdinLock<'a> {
    inner: MutexGuard<'a, StdioReader<StdinRaw>>,
}

/// Constructs a new handle to the standard input of the current process.
///
/// Each handle returned is a reference to a shared global buffer whose access
/// is synchronized via a mutex. If you need more explicit control over
/// locking, see the [`Stdin::lock`] method.
///
/// [`Stdin::lock`]: struct.Stdin.html#method.lock
///
/// ### Note: Windows Portability Consideration
/// When operating in a console, the Windows implementation of this stream does not support
/// non-UTF-8 byte sequences. Attempting to read bytes that are not valid UTF-8 will return
/// an error.
///
/// # Examples
///
/// Using implicit synchronization:
///
/// ```no_run
/// use std::io::{self, Read};
///
/// fn main() -> io::Result<()> {
///     let mut buffer = String::new();
///     io::stdin().read_to_string(&mut buffer)?;
///     Ok(())
/// }
/// ```
///
/// Using explicit synchronization:
///
/// ```no_run
/// use std::io::{self, Read};
///
/// fn main() -> io::Result<()> {
///     let mut buffer = String::new();
///     let stdin = io::stdin();
///     let mut handle = stdin.lock();
///
///     handle.read_to_string(&mut buffer)?;
///     Ok(())
/// }
/// ```
#[stable(feature = "rust1", since = "1.0.0")]
pub fn stdin() -> Stdin {
    static INSTANCE: Lazy<Mutex<StdioReader<StdinRaw>>> = Lazy::new();
    return Stdin {
        inner: unsafe {
            INSTANCE.get(stdin_init).expect("cannot access stdin during shutdown")
        },
    };

    fn stdin_init() -> Arc<Mutex<StdioReader<StdinRaw>>> {
        // This must not reentrantly access `INSTANCE`
        let stdin = match stdin_raw() {
            Ok(stdin) => StdioReader::new_buffered(stdin),
            _ => StdioReader::new_fake(),
        };

        Arc::new(Mutex::new(stdin))
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
    /// ```no_run
    /// use std::io::{self, Read};
    ///
    /// fn main() -> io::Result<()> {
    ///     let mut buffer = String::new();
    ///     let stdin = io::stdin();
    ///     let mut handle = stdin.lock();
    ///
    ///     handle.read_to_string(&mut buffer)?;
    ///     Ok(())
    /// }
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn lock(&self) -> StdinLock<'_> {
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
    /// - Pipe some text to it, e.g., `printf foo | path/to/executable`
    /// - Give it text interactively by running the executable directly,
    ///   in which case it will wait for the Enter key to be pressed before
    ///   continuing
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn read_line(&self, buf: &mut String) -> io::Result<usize> {
        self.lock().read_line(buf)
    }
}

#[stable(feature = "std_debug", since = "1.16.0")]
impl fmt::Debug for Stdin {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.pad("Stdin { .. }")
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl Read for Stdin {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        self.lock().read(buf)
    }
    fn read_vectored(&mut self, bufs: &mut [IoSliceMut<'_>]) -> io::Result<usize> {
        self.lock().read_vectored(bufs)
    }
    #[inline]
    unsafe fn initializer(&self) -> Initializer {
        Initializer::nop()
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
impl Read for StdinLock<'_> {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        self.inner.read(buf)
    }

    fn read_vectored(&mut self, bufs: &mut [IoSliceMut<'_>]) -> io::Result<usize> {
        self.inner.read_vectored(bufs)
    }

    #[inline]
    unsafe fn initializer(&self) -> Initializer {
        Initializer::nop()
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl BufRead for StdinLock<'_> {
    fn fill_buf(&mut self) -> io::Result<&[u8]> { self.inner.fill_buf() }
    fn consume(&mut self, n: usize) { self.inner.consume(n) }
}

#[stable(feature = "std_debug", since = "1.16.0")]
impl fmt::Debug for StdinLock<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.pad("StdinLock { .. }")
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
/// ### Note: Windows Portability Consideration
/// When operating in a console, the Windows implementation of this stream does not support
/// non-UTF-8 byte sequences. Attempting to write bytes that are not valid UTF-8 will return
/// an error.
///
/// [`lock`]: #method.lock
/// [`io::stdout`]: fn.stdout.html
#[stable(feature = "rust1", since = "1.0.0")]
pub struct Stdout {
    inner: Arc<ReentrantMutex<RefCell<StdioWriter<StdoutRaw>>>>,
}

/// A locked reference to the `Stdout` handle.
///
/// This handle implements the [`Write`] trait, and is constructed via
/// the [`Stdout::lock`] method.
///
/// ### Note: Windows Portability Consideration
/// When operating in a console, the Windows implementation of this stream does not support
/// non-UTF-8 byte sequences. Attempting to write bytes that are not valid UTF-8 will return
/// an error.
///
/// [`Write`]: trait.Write.html
/// [`Stdout::lock`]: struct.Stdout.html#method.lock
#[stable(feature = "rust1", since = "1.0.0")]
pub struct StdoutLock<'a> {
    inner: ReentrantMutexGuard<'a, RefCell<StdioWriter<StdoutRaw>>>,
}

/// Constructs a new handle to the standard output of the current process.
///
/// Each handle returned is a reference to a shared global buffer whose access
/// is synchronized via a mutex. If you need more explicit control over
/// locking, see the [`Stdout::lock`] method.
///
/// [`Stdout::lock`]: struct.Stdout.html#method.lock
///
/// ### Note: Windows Portability Consideration
/// When operating in a console, the Windows implementation of this stream does not support
/// non-UTF-8 byte sequences. Attempting to write bytes that are not valid UTF-8 will return
/// an error.
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
#[stable(feature = "rust1", since = "1.0.0")]
pub fn stdout() -> Stdout {
    static INSTANCE: Lazy<ReentrantMutex<RefCell<StdioWriter<StdoutRaw>>>> = Lazy::new();
    return Stdout {
        inner: unsafe {
            INSTANCE.get(stdout_init).expect("cannot access stdout during shutdown")
        },
    };

    fn stdout_init() -> Arc<ReentrantMutex<RefCell<StdioWriter<StdoutRaw>>>> {
        // This must not reentrantly access `INSTANCE`
        let stdout = match stdout_raw() {
            Ok(stdout) => {
                let buffering = if stdout.should_be_line_buffered() {
                    StdioBufferKind::LineBuffered
                } else {
                    StdioBufferKind::Buffered
                };
                StdioWriter::new(stdout, buffering)
            },
            _ => StdioWriter::new_fake(),
        };
        Arc::new(ReentrantMutex::new(RefCell::new(stdout)))
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
    ///     let stdout = io::stdout();
    ///     let mut handle = stdout.lock();
    ///
    ///     handle.write_all(b"hello world")?;
    ///
    ///     Ok(())
    /// }
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn lock(&self) -> StdoutLock<'_> {
        StdoutLock { inner: self.inner.lock().unwrap_or_else(|e| e.into_inner()) }
    }
}

#[stable(feature = "std_debug", since = "1.16.0")]
impl fmt::Debug for Stdout {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.pad("Stdout { .. }")
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl Write for Stdout {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        self.lock().write(buf)
    }
    fn write_vectored(&mut self, bufs: &[IoSlice<'_>]) -> io::Result<usize> {
        self.lock().write_vectored(bufs)
    }
    fn flush(&mut self) -> io::Result<()> {
        self.lock().flush()
    }
    fn write_all(&mut self, buf: &[u8]) -> io::Result<()> {
        self.lock().write_all(buf)
    }
    fn write_fmt(&mut self, args: fmt::Arguments<'_>) -> io::Result<()> {
        self.lock().write_fmt(args)
    }
}
#[stable(feature = "rust1", since = "1.0.0")]
impl Write for StdoutLock<'_> {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        self.inner.borrow_mut().write(buf)
    }
    fn write_vectored(&mut self, bufs: &[IoSlice<'_>]) -> io::Result<usize> {
        self.inner.borrow_mut().write_vectored(bufs)
    }
    fn flush(&mut self) -> io::Result<()> {
        self.inner.borrow_mut().flush()
    }
}

#[stable(feature = "std_debug", since = "1.16.0")]
impl fmt::Debug for StdoutLock<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.pad("StdoutLock { .. }")
    }
}

/// A handle to the standard error stream of a process.
///
/// For more information, see the [`io::stderr`] method.
///
/// [`io::stderr`]: fn.stderr.html
///
/// ### Note: Windows Portability Consideration
/// When operating in a console, the Windows implementation of this stream does not support
/// non-UTF-8 byte sequences. Attempting to write bytes that are not valid UTF-8 will return
/// an error.
#[stable(feature = "rust1", since = "1.0.0")]
pub struct Stderr {
    inner: Arc<ReentrantMutex<RefCell<StdioWriter<StderrRaw>>>>,
}

/// A locked reference to the `Stderr` handle.
///
/// This handle implements the `Write` trait and is constructed via
/// the [`Stderr::lock`] method.
///
/// [`Stderr::lock`]: struct.Stderr.html#method.lock
///
/// ### Note: Windows Portability Consideration
/// When operating in a console, the Windows implementation of this stream does not support
/// non-UTF-8 byte sequences. Attempting to write bytes that are not valid UTF-8 will return
/// an error.
#[stable(feature = "rust1", since = "1.0.0")]
pub struct StderrLock<'a> {
    inner: ReentrantMutexGuard<'a, RefCell<StdioWriter<StderrRaw>>>,
}

/// Constructs a new handle to the standard error of the current process.
///
/// This handle is not buffered.
///
/// ### Note: Windows Portability Consideration
/// When operating in a console, the Windows implementation of this stream does not support
/// non-UTF-8 byte sequences. Attempting to write bytes that are not valid UTF-8 will return
/// an error.
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
#[stable(feature = "rust1", since = "1.0.0")]
pub fn stderr() -> Stderr {
    static INSTANCE: Lazy<ReentrantMutex<RefCell<StdioWriter<StderrRaw>>>> = Lazy::new();
    return Stderr {
        inner: unsafe {
            INSTANCE.get(stderr_init).expect("cannot access stderr during shutdown")
        },
    };

    fn stderr_init() -> Arc<ReentrantMutex<RefCell<StdioWriter<StderrRaw>>>> {
        // This must not reentrantly access `INSTANCE`
        let stderr = match stderr_raw() {
            Ok(stderr) => StdioWriter::new(stderr, StdioBufferKind::Unbuffered),
            _ => StdioWriter::new_fake(),
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
    ///     handle.write_all(b"hello world")?;
    ///
    ///     Ok(())
    /// }
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn lock(&self) -> StderrLock<'_> {
        StderrLock { inner: self.inner.lock().unwrap_or_else(|e| e.into_inner()) }
    }
}

#[stable(feature = "std_debug", since = "1.16.0")]
impl fmt::Debug for Stderr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.pad("Stderr { .. }")
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl Write for Stderr {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        self.lock().write(buf)
    }
    fn write_vectored(&mut self, bufs: &[IoSlice<'_>]) -> io::Result<usize> {
        self.lock().write_vectored(bufs)
    }
    fn flush(&mut self) -> io::Result<()> {
        self.lock().flush()
    }
    fn write_all(&mut self, buf: &[u8]) -> io::Result<()> {
        self.lock().write_all(buf)
    }
    fn write_fmt(&mut self, args: fmt::Arguments<'_>) -> io::Result<()> {
        self.lock().write_fmt(args)
    }
}
#[stable(feature = "rust1", since = "1.0.0")]
impl Write for StderrLock<'_> {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        self.inner.borrow_mut().write(buf)
    }
    fn write_vectored(&mut self, bufs: &[IoSlice<'_>]) -> io::Result<usize> {
        self.inner.borrow_mut().write_vectored(bufs)
    }
    fn flush(&mut self) -> io::Result<()> {
        self.inner.borrow_mut().flush()
    }
}

#[stable(feature = "std_debug", since = "1.16.0")]
impl fmt::Debug for StderrLock<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
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
pub fn set_panic(sink: Option<Box<dyn Write + Send>>) -> Option<Box<dyn Write + Send>> {
    use crate::mem;
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
pub fn set_print(sink: Option<Box<dyn Write + Send>>) -> Option<Box<dyn Write + Send>> {
    use crate::mem;
    LOCAL_STDOUT.with(move |slot| {
        mem::replace(&mut *slot.borrow_mut(), sink)
    }).and_then(|mut s| {
        let _ = s.flush();
        Some(s)
    })
}

/// Write `args` to output stream `local_s` if possible, `global_s`
/// otherwise. `label` identifies the stream in a panic message.
///
/// This function is used to print error messages, so it takes extra
/// care to avoid causing a panic when `local_stream` is unusable.
/// For instance, if the TLS key for the local stream is
/// already destroyed, or if the local stream is locked by another
/// thread, it will just fall back to the global stream.
///
/// However, if the actual I/O causes an error, this function does panic.
fn print_to<T>(
    args: fmt::Arguments<'_>,
    local_s: &'static LocalKey<RefCell<Option<Box<dyn Write+Send>>>>,
    global_s: fn() -> T,
    label: &str,
)
where
    T: Write,
{
    let result = local_s.try_with(|s| {
        if let Ok(mut borrowed) = s.try_borrow_mut() {
            if let Some(w) = borrowed.as_mut() {
                return w.write_fmt(args);
            }
        }
        global_s().write_fmt(args)
    }).unwrap_or_else(|_| {
        global_s().write_fmt(args)
    });

    if let Err(e) = result {
        panic!("failed printing to {}: {}", label, e);
    }
}

#[unstable(feature = "print_internals",
           reason = "implementation detail which may disappear or be replaced at any time",
           issue = "0")]
#[doc(hidden)]
#[cfg(not(test))]
pub fn _print(args: fmt::Arguments<'_>) {
    print_to(args, &LOCAL_STDOUT, stdout, "stdout");
}

#[unstable(feature = "print_internals",
           reason = "implementation detail which may disappear or be replaced at any time",
           issue = "0")]
#[doc(hidden)]
#[cfg(not(test))]
pub fn _eprint(args: fmt::Arguments<'_>) {
    print_to(args, &LOCAL_STDERR, stderr, "stderr");
}

#[cfg(test)]
pub use realstd::io::{_eprint, _print};

#[cfg(test)]
mod tests {
    use crate::panic::{UnwindSafe, RefUnwindSafe};
    use crate::thread;
    use super::*;

    #[test]
    fn stdout_unwind_safe() {
        assert_unwind_safe::<Stdout>();
    }
    #[test]
    fn stdoutlock_unwind_safe() {
        assert_unwind_safe::<StdoutLock<'_>>();
        assert_unwind_safe::<StdoutLock<'static>>();
    }
    #[test]
    fn stderr_unwind_safe() {
        assert_unwind_safe::<Stderr>();
    }
    #[test]
    fn stderrlock_unwind_safe() {
        assert_unwind_safe::<StderrLock<'_>>();
        assert_unwind_safe::<StderrLock<'static>>();
    }

    fn assert_unwind_safe<T: UnwindSafe + RefUnwindSafe>() {}

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
