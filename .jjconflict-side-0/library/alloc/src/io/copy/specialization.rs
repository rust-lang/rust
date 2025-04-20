//! Provides specialization for `io::copy`.

use super::CopyState;
use crate::io::{BufReader, Read, Result, Take, Write};

pub(super) trait SpecCopyInner {
    /// The implementation of `io::copy` that can rely on platform specific specialization
    /// provided by `libstd`.
    fn copy<W: Write + ?Sized>(&mut self, writer: &mut W) -> Result<CopyState>;
}

impl<R: Read + ?Sized> SpecCopyInner for R {
    default fn copy<W: Write + ?Sized>(&mut self, _writer: &mut W) -> Result<CopyState> {
        Ok(CopyState::Fallback(0))
    }
}

impl<R: SpecCopy> SpecCopyInner for R {
    fn copy<W: Write + ?Sized>(&mut self, writer: &mut W) -> Result<CopyState> {
        <R as SpecCopy>::copy(self, writer)
    }
}

#[doc(hidden)]
#[unstable(feature = "core_io_internals", reason = "exposed only for libstd", issue = "none")]
#[rustc_specialization_trait]
pub trait SpecCopy: Read {
    /// Attempt to copy from this reader to the provided writer using a specialized
    /// process.
    ///
    /// Note that this function does _not_ take `self` as a parameter, and instead
    /// is passed a generic `Read` type `R`.
    /// This allows the `Self` type to provide specialized implementations for
    /// any combination of `Read` and `Write` types.
    /// However, in practice `Self` and types wrapping `Self` will be passed as
    /// the `reader` argument.
    ///
    /// As of time of writing, `&mut R`, `Take<R>`, and `BufReader<R>` will
    /// forward to `R` for a specialized copy implementation.
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
