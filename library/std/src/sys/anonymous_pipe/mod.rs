//! Module for annoymous pipe
//!
//! ```
//! #![feature(anonymous_pipe)]
//! # fn main() -> std::io::Result<()> {
//! let (reader, writer) = std::pipe::pipe()?;
//! # Ok(())
//! # }
//! ```

use crate::{io, sys::pipe::AnonPipe};

/// Create annoymous pipe that is close-on-exec and blocking.
#[unstable(feature = "anonymous_pipe", issue = "127154")]
#[inline]
pub fn pipe() -> io::Result<(PipeReader, PipeWriter)> {
    cfg_if::cfg_if! {
        if #[cfg(unix)] {
            unix::pipe()
        } else if #[cfg(windows)] {
            windows::pipe()
        } else {
            Err(io::Error::UNSUPPORTED_PLATFORM)
        }
    }
}

/// Read end of the annoymous pipe.
#[unstable(feature = "anonymous_pipe", issue = "127154")]
#[derive(Debug)]
pub struct PipeReader(AnonPipe);

/// Write end of the annoymous pipe.
#[unstable(feature = "anonymous_pipe", issue = "127154")]
#[derive(Debug)]
pub struct PipeWriter(AnonPipe);

impl PipeReader {
    /// Create a new [`PipeReader`] instance that shares the same underlying file description.
    #[unstable(feature = "anonymous_pipe", issue = "127154")]
    pub fn try_clone(&self) -> io::Result<Self> {
        self.0.try_clone().map(Self)
    }
}

impl PipeWriter {
    /// Create a new [`PipeWriter`] instance that shares the same underlying file description.
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

#[cfg(unix)]
mod unix;

#[cfg(windows)]
mod windows;

#[cfg(test)]
mod tests;
