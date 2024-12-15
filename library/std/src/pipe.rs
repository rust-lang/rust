//!  A cross-platform anonymous pipe.
//!
//! This module provides support for anonymous OS pipes, like [pipe] on Linux or [CreatePipe] on
//! Windows.
//!
//! # Behavior
//!
//! A pipe is a synchronous, unidirectional data channel between two or more processes, like an
//! interprocess [`mpsc`](crate::sync::mpsc) provided by the OS. In particular:
//!
//! * A read on a [`PipeReader`] blocks until the pipe is non-empty.
//! * A write on a [`PipeWriter`] blocks when the pipe is full.
//! * When all copies of a [`PipeWriter`] are closed, a read on the corresponding [`PipeReader`]
//!   returns EOF.
//! * [`PipeReader`] can be shared, but only one process will consume the data in the pipe.
//!
//! # Capacity
//!
//! Pipe capacity is platform dependent. To quote the Linux [man page]:
//!
//! > Different implementations have different limits for the pipe capacity. Applications should
//! > not rely on a particular capacity: an application should be designed so that a reading process
//! > consumes data as soon as it is available, so that a writing process does not remain blocked.
//!
//! # Examples
//!
//! ```no_run
//! #![feature(anonymous_pipe)]
//! # #[cfg(miri)] fn main() {}
//! # #[cfg(not(miri))]
//! # fn main() -> std::io::Result<()> {
//! # use std::process::Command;
//! # use std::io::{Read, Write};
//! let (ping_rx, mut ping_tx) = std::pipe::pipe()?;
//! let (mut pong_rx, pong_tx) = std::pipe::pipe()?;
//!
//! // Spawn a process that echoes its input.
//! let mut echo_server = Command::new("cat").stdin(ping_rx).stdout(pong_tx).spawn()?;
//!
//! ping_tx.write_all(b"hello")?;
//! // Close to unblock echo_server's reader.
//! drop(ping_tx);
//!
//! let mut buf = String::new();
//! // Block until echo_server's writer is closed.
//! pong_rx.read_to_string(&mut buf)?;
//! assert_eq!(&buf, "hello");
//!
//! echo_server.wait()?;
//! # Ok(())
//! # }
//! ```
//! [pipe]: https://man7.org/linux/man-pages/man2/pipe.2.html
//! [CreatePipe]: https://learn.microsoft.com/en-us/windows/win32/api/namedpipeapi/nf-namedpipeapi-createpipe
//! [man page]: https://man7.org/linux/man-pages/man7/pipe.7.html
use crate::io;
use crate::sys::anonymous_pipe::{AnonPipe, pipe as pipe_inner};

/// Create anonymous pipe that is close-on-exec and blocking.
///
/// # Examples
///
/// See the [module-level](crate::pipe) documentation for examples.
#[unstable(feature = "anonymous_pipe", issue = "127154")]
#[inline]
pub fn pipe() -> io::Result<(PipeReader, PipeWriter)> {
    pipe_inner().map(|(reader, writer)| (PipeReader(reader), PipeWriter(writer)))
}

/// Read end of the anonymous pipe.
#[unstable(feature = "anonymous_pipe", issue = "127154")]
#[derive(Debug)]
pub struct PipeReader(pub(crate) AnonPipe);

/// Write end of the anonymous pipe.
#[unstable(feature = "anonymous_pipe", issue = "127154")]
#[derive(Debug)]
pub struct PipeWriter(pub(crate) AnonPipe);

impl PipeReader {
    /// Create a new [`PipeReader`] instance that shares the same underlying file description.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// #![feature(anonymous_pipe)]
    /// # #[cfg(miri)] fn main() {}
    /// # #[cfg(not(miri))]
    /// # fn main() -> std::io::Result<()> {
    /// # use std::fs;
    /// # use std::io::Write;
    /// # use std::process::Command;
    /// const NUM_SLOT: u8 = 2;
    /// const NUM_PROC: u8 = 5;
    /// const OUTPUT: &str = "work.txt";
    ///
    /// let mut jobs = vec![];
    /// let (reader, mut writer) = std::pipe::pipe()?;
    ///
    /// // Write NUM_SLOT characters the the pipe.
    /// writer.write_all(&[b'|'; NUM_SLOT as usize])?;
    ///
    /// // Spawn several processes that read a character from the pipe, do some work, then
    /// // write back to the pipe. When the pipe is empty, the processes block, so only
    /// // NUM_SLOT processes can be working at any given time.
    /// for _ in 0..NUM_PROC {
    ///     jobs.push(
    ///         Command::new("bash")
    ///             .args(["-c",
    ///                 &format!(
    ///                      "read -n 1\n\
    ///                       echo -n 'x' >> '{OUTPUT}'\n\
    ///                       echo -n '|'",
    ///                 ),
    ///             ])
    ///             .stdin(reader.try_clone()?)
    ///             .stdout(writer.try_clone()?)
    ///             .spawn()?,
    ///     );
    /// }
    ///
    /// // Wait for all jobs to finish.
    /// for mut job in jobs {
    ///     job.wait()?;
    /// }
    ///
    /// // Check our work and clean up.
    /// let xs = fs::read_to_string(OUTPUT)?;
    /// fs::remove_file(OUTPUT)?;
    /// assert_eq!(xs, "x".repeat(NUM_PROC.into()));
    /// # Ok(())
    /// # }
    /// ```
    #[unstable(feature = "anonymous_pipe", issue = "127154")]
    pub fn try_clone(&self) -> io::Result<Self> {
        self.0.try_clone().map(Self)
    }
}

impl PipeWriter {
    /// Create a new [`PipeWriter`] instance that shares the same underlying file description.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// #![feature(anonymous_pipe)]
    /// # #[cfg(miri)] fn main() {}
    /// # #[cfg(not(miri))]
    /// # fn main() -> std::io::Result<()> {
    /// # use std::process::Command;
    /// # use std::io::Read;
    /// let (mut reader, writer) = std::pipe::pipe()?;
    ///
    /// // Spawn a process that writes to stdout and stderr.
    /// let mut peer = Command::new("bash")
    ///     .args([
    ///         "-c",
    ///         "echo -n foo\n\
    ///          echo -n bar >&2"
    ///     ])
    ///     .stdout(writer.try_clone()?)
    ///     .stderr(writer)
    ///     .spawn()?;
    ///
    /// // Read and check the result.
    /// let mut msg = String::new();
    /// reader.read_to_string(&mut msg)?;
    /// assert_eq!(&msg, "foobar");
    ///
    /// peer.wait()?;
    /// # Ok(())
    /// # }
    /// ```
    #[unstable(feature = "anonymous_pipe", issue = "127154")]
    pub fn try_clone(&self) -> io::Result<Self> {
        self.0.try_clone().map(Self)
    }
}

#[unstable(feature = "anonymous_pipe", issue = "127154")]
impl io::Read for &PipeReader {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        self.0.read(buf)
    }
    fn read_vectored(&mut self, bufs: &mut [io::IoSliceMut<'_>]) -> io::Result<usize> {
        self.0.read_vectored(bufs)
    }
    #[inline]
    fn is_read_vectored(&self) -> bool {
        self.0.is_read_vectored()
    }
    fn read_to_end(&mut self, buf: &mut Vec<u8>) -> io::Result<usize> {
        self.0.read_to_end(buf)
    }
    fn read_buf(&mut self, buf: io::BorrowedCursor<'_>) -> io::Result<()> {
        self.0.read_buf(buf)
    }
}

#[unstable(feature = "anonymous_pipe", issue = "127154")]
impl io::Read for PipeReader {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        self.0.read(buf)
    }
    fn read_vectored(&mut self, bufs: &mut [io::IoSliceMut<'_>]) -> io::Result<usize> {
        self.0.read_vectored(bufs)
    }
    #[inline]
    fn is_read_vectored(&self) -> bool {
        self.0.is_read_vectored()
    }
    fn read_to_end(&mut self, buf: &mut Vec<u8>) -> io::Result<usize> {
        self.0.read_to_end(buf)
    }
    fn read_buf(&mut self, buf: io::BorrowedCursor<'_>) -> io::Result<()> {
        self.0.read_buf(buf)
    }
}

#[unstable(feature = "anonymous_pipe", issue = "127154")]
impl io::Write for &PipeWriter {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        self.0.write(buf)
    }
    #[inline]
    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }

    fn write_vectored(&mut self, bufs: &[io::IoSlice<'_>]) -> io::Result<usize> {
        self.0.write_vectored(bufs)
    }

    #[inline]
    fn is_write_vectored(&self) -> bool {
        self.0.is_write_vectored()
    }
}

#[unstable(feature = "anonymous_pipe", issue = "127154")]
impl io::Write for PipeWriter {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        self.0.write(buf)
    }
    #[inline]
    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }

    fn write_vectored(&mut self, bufs: &[io::IoSlice<'_>]) -> io::Result<usize> {
        self.0.write_vectored(bufs)
    }

    #[inline]
    fn is_write_vectored(&self) -> bool {
        self.0.is_write_vectored()
    }
}
