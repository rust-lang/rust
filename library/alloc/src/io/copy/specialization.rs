//! Provides specialization for `io::copy`.

use super::CopyState;
use crate::io::{BufReader, Read, Result, Take, Write};

/// The implementation of `io::copy` that can rely on platform specific specialization
/// provided by `libstd`.
pub(super) fn specialized_copy<R: ?Sized, W: ?Sized>(
    reader: &mut R,
    writer: &mut W,
) -> Result<CopyState>
where
    R: Read,
    W: Write,
{
    SpecCopyInner::copy((reader, writer))
}

trait SpecCopyInner {
    fn copy(self) -> Result<CopyState>;
}

impl<R: Read + ?Sized, W: Write + ?Sized> SpecCopyInner for (&mut R, &mut W) {
    default fn copy(self) -> Result<CopyState> {
        Ok(CopyState::Fallback(0))
    }
}

impl<R: SpecCopy, W: Write> SpecCopyInner for (&mut R, &mut W) {
    fn copy(self) -> Result<CopyState> {
        <R as SpecCopy>::copy(self.0, self.1)
    }
}

#[doc(hidden)]
#[unstable(feature = "core_io_internals", reason = "exposed only for libstd", issue = "none")]
#[rustc_specialization_trait]
pub trait SpecCopy: Read {
    /// Attempt to copy from this reader to the provided writer using a specialized
    /// process.
    fn copy<R: Read + ?Sized, W: Write + ?Sized>(
        _reader: &mut R,
        _writer: &mut W,
    ) -> Result<CopyState>;
}

impl<T> SpecCopy for &mut T
where
    T: SpecCopy,
{
    fn copy<R: Read + ?Sized, W: Write + ?Sized>(
        reader: &mut R,
        writer: &mut W,
    ) -> Result<CopyState> {
        <T as SpecCopy>::copy(reader, writer)
    }
}

impl<T: SpecCopy> SpecCopy for Take<T> {
    fn copy<R: Read + ?Sized, W: Write + ?Sized>(
        reader: &mut R,
        writer: &mut W,
    ) -> Result<CopyState> {
        <T as SpecCopy>::copy(reader, writer)
    }
}

impl<T: ?Sized + SpecCopy> SpecCopy for BufReader<T> {
    fn copy<R: Read + ?Sized, W: Write + ?Sized>(
        reader: &mut R,
        writer: &mut W,
    ) -> Result<CopyState> {
        <T as SpecCopy>::copy(reader, writer)
    }
}
