#[cfg(test)]
mod tests;

use crate::alloc::Allocator;
use crate::collections::VecDeque;
use crate::io::{self, BufRead};

// =============================================================================
// Forwarding implementations

#[stable(feature = "rust1", since = "1.0.0")]
impl<B: BufRead + ?Sized> BufRead for &mut B {
    #[inline]
    fn fill_buf(&mut self) -> io::Result<&[u8]> {
        (**self).fill_buf()
    }

    #[inline]
    fn consume(&mut self, amt: usize) {
        (**self).consume(amt)
    }

    #[inline]
    fn has_data_left(&mut self) -> io::Result<bool> {
        (**self).has_data_left()
    }

    #[inline]
    fn read_until(&mut self, byte: u8, buf: &mut Vec<u8>) -> io::Result<usize> {
        (**self).read_until(byte, buf)
    }

    #[inline]
    fn skip_until(&mut self, byte: u8) -> io::Result<usize> {
        (**self).skip_until(byte)
    }

    #[inline]
    fn read_line(&mut self, buf: &mut String) -> io::Result<usize> {
        (**self).read_line(buf)
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<B: BufRead + ?Sized> BufRead for Box<B> {
    #[inline]
    fn fill_buf(&mut self) -> io::Result<&[u8]> {
        (**self).fill_buf()
    }

    #[inline]
    fn consume(&mut self, amt: usize) {
        (**self).consume(amt)
    }

    #[inline]
    fn has_data_left(&mut self) -> io::Result<bool> {
        (**self).has_data_left()
    }

    #[inline]
    fn read_until(&mut self, byte: u8, buf: &mut Vec<u8>) -> io::Result<usize> {
        (**self).read_until(byte, buf)
    }

    #[inline]
    fn skip_until(&mut self, byte: u8) -> io::Result<usize> {
        (**self).skip_until(byte)
    }

    #[inline]
    fn read_line(&mut self, buf: &mut String) -> io::Result<usize> {
        (**self).read_line(buf)
    }
}

// =============================================================================
// In-memory buffer implementations

#[stable(feature = "rust1", since = "1.0.0")]
impl BufRead for &[u8] {
    #[inline]
    fn fill_buf(&mut self) -> io::Result<&[u8]> {
        Ok(*self)
    }

    #[inline]
    fn consume(&mut self, amt: usize) {
        *self = &self[amt..];
    }
}

/// BufRead is implemented for `VecDeque<u8>` by reading bytes from the front of the `VecDeque`.
#[stable(feature = "vecdeque_buf_read", since = "1.75.0")]
impl<A: Allocator> BufRead for VecDeque<u8, A> {
    /// Returns the contents of the "front" slice as returned by
    /// [`as_slices`][`VecDeque::as_slices`]. If the contained byte slices of the `VecDeque` are
    /// discontiguous, multiple calls to `fill_buf` will be needed to read the entire content.
    #[inline]
    fn fill_buf(&mut self) -> io::Result<&[u8]> {
        let (front, _) = self.as_slices();
        Ok(front)
    }

    #[inline]
    fn consume(&mut self, amt: usize) {
        self.drain(..amt);
    }
}
