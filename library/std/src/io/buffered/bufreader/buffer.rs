use crate::cmp;
use crate::io::{self, Read, ReadBuf};
use crate::mem::MaybeUninit;

pub struct Buffer {
    buf: Box<[MaybeUninit<u8>]>,
    pos: usize,
    cap: usize,
    init: usize,
}

impl Buffer {
    #[inline]
    pub fn with_capacity(capacity: usize) -> Self {
        let buf = Box::new_uninit_slice(capacity);
        Self { buf, pos: 0, cap: 0, init: 0 }
    }

    #[inline]
    pub fn buffer(&self) -> &[u8] {
        // SAFETY: self.cap is always <= self.init, so self.buf[self.pos..self.cap] is always init
        // Additionally, both self.pos and self.cap are valid and and self.cap => self.pos, and
        // that region is initialized because those are all invariants of this type.
        unsafe { MaybeUninit::slice_assume_init_ref(&self.buf.get_unchecked(self.pos..self.cap)) }
    }

    #[inline]
    pub fn capacity(&self) -> usize {
        self.buf.len()
    }

    #[inline]
    pub fn cap(&self) -> usize {
        self.cap
    }

    #[inline]
    pub fn pos(&self) -> usize {
        self.pos
    }

    #[inline]
    pub fn discard_buffer(&mut self) {
        self.pos = 0;
        self.cap = 0;
    }

    #[inline]
    pub fn consume(&mut self, amt: usize) {
        self.pos = cmp::min(self.pos + amt, self.cap);
    }

    #[inline]
    pub fn unconsume(&mut self, amt: usize) {
        self.pos = self.pos.saturating_sub(amt);
    }

    #[inline]
    pub fn fill_buf(&mut self, mut reader: impl Read) -> io::Result<&[u8]> {
        // If we've reached the end of our internal buffer then we need to fetch
        // some more data from the underlying reader.
        // Branch using `>=` instead of the more correct `==`
        // to tell the compiler that the pos..cap slice is always valid.
        if self.pos >= self.cap {
            debug_assert!(self.pos == self.cap);

            let mut readbuf = ReadBuf::uninit(&mut self.buf);

            // SAFETY: `self.init` is either 0 or set to `readbuf.initialized_len()`
            // from the last time this function was called
            unsafe {
                readbuf.assume_init(self.init);
            }

            reader.read_buf(&mut readbuf)?;

            self.cap = readbuf.filled_len();
            self.init = readbuf.initialized_len();

            self.pos = 0;
        }
        Ok(self.buffer())
    }
}
