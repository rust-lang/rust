use super::err2io;
use crate::io::{default_read, BorrowedCursor, Read, Result};

pub const INSECURE_HASHMAP: bool = false;

pub struct Entropy {
    pub insecure: bool,
}

impl Read for Entropy {
    #[inline]
    fn read(&mut self, buf: &mut [u8]) -> Result<usize> {
        default_read(self, buf)
    }

    #[inline]
    fn read_buf(&mut self, mut buf: BorrowedCursor<'_>) -> Result<()> {
        unsafe { wasi::random_get(buf.as_ptr(), buf.capacity()).map_err(err2io) }
    }

    #[inline]
    fn read_buf_exact(&mut self, buf: BorrowedCursor<'_>) -> Result<()> {
        self.read_buf(buf)
    }
}
