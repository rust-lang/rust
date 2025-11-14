//! An encapsulation of `BufReader`'s buffer management logic.
//!
//! This module factors out the basic functionality of `BufReader` in order to protect two core
//! invariants:
//! * `filled` bytes of `buf` are always initialized
//! * `pos` is always <= `filled`
//! Since this module encapsulates the buffer management logic, we can ensure that the range
//! `pos..filled` is always a valid index into the initialized region of the buffer. This means
//! that user code which wants to do reads from a `BufReader` via `buffer` + `consume` can do so
//! without encountering any runtime bounds checks.

use crate::cmp;
use crate::io::{self, BorrowedBuf, ErrorKind, Read};
use crate::mem::MaybeUninit;

pub struct Buffer {
    // The buffer.
    buf: Box<[MaybeUninit<u8>]>,
    // The current seek offset into `buf`, must always be <= `filled`.
    pos: usize,
    // Each call to `fill_buf` sets `filled` to indicate how many bytes at the start of `buf` are
    // initialized with bytes from a read.
    filled: usize,
}

impl Buffer {
    #[inline]
    pub fn with_capacity(capacity: usize) -> Self {
        let buf = Box::new_uninit_slice(capacity);
        Self { buf, pos: 0, filled: 0 }
    }

    #[inline]
    pub fn try_with_capacity(capacity: usize) -> io::Result<Self> {
        match Box::try_new_uninit_slice(capacity) {
            Ok(buf) => Ok(Self { buf, pos: 0, filled: 0 }),
            Err(_) => {
                Err(io::const_error!(ErrorKind::OutOfMemory, "failed to allocate read buffer"))
            }
        }
    }

    #[inline]
    pub fn buffer(&self) -> &[u8] {
        // SAFETY: self.pos and self.cap are valid, and self.cap => self.pos, and
        // that region is initialized because those are all invariants of this type.
        unsafe { self.buf.get_unchecked(self.pos..self.filled).assume_init_ref() }
    }

    #[inline]
    pub fn capacity(&self) -> usize {
        self.buf.len()
    }

    #[inline]
    pub fn filled(&self) -> usize {
        self.filled
    }

    #[inline]
    pub fn pos(&self) -> usize {
        self.pos
    }

    #[inline]
    pub fn discard_buffer(&mut self) {
        self.pos = 0;
        self.filled = 0;
    }

    #[inline]
    pub fn consume(&mut self, amt: usize) {
        self.pos = cmp::min(self.pos + amt, self.filled);
    }

    /// If there are `amt` bytes available in the buffer, pass a slice containing those bytes to
    /// `visitor` and return true. If there are not enough bytes available, return false.
    #[inline]
    pub fn consume_with<V>(&mut self, amt: usize, mut visitor: V) -> bool
    where
        V: FnMut(&[u8]),
    {
        if let Some(claimed) = self.buffer().get(..amt) {
            visitor(claimed);
            // If the indexing into self.buffer() succeeds, amt must be a valid increment.
            self.pos += amt;
            true
        } else {
            false
        }
    }

    #[inline]
    pub fn unconsume(&mut self, amt: usize) {
        self.pos = self.pos.saturating_sub(amt);
    }

    /// Read more bytes into the buffer without discarding any of its contents
    pub fn read_more(&mut self, mut reader: impl Read) -> io::Result<usize> {
        let mut buf = BorrowedBuf::from(&mut self.buf[self.filled..]);
        reader.read_buf(buf.unfilled())?;
        self.filled += buf.len();
        Ok(buf.len())
    }

    /// Remove bytes that have already been read from the buffer.
    pub fn backshift(&mut self) {
        self.buf.copy_within(self.pos..self.filled, 0);
        self.filled -= self.pos;
        self.pos = 0;
    }

    #[inline]
    pub fn fill_buf(&mut self, mut reader: impl Read) -> io::Result<&[u8]> {
        // If we've reached the end of our internal buffer then we need to fetch
        // some more data from the reader.
        // Branch using `>=` instead of the more correct `==`
        // to tell the compiler that the pos..cap slice is always valid.
        if self.pos >= self.filled {
            debug_assert!(self.pos == self.filled);

            let mut buf = BorrowedBuf::from(&mut *self.buf);
            let result = reader.read_buf(buf.unfilled());

            self.pos = 0;
            self.filled = buf.len();

            result?;
        }
        Ok(self.buffer())
    }
}
