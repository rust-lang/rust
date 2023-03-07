use super::{abi, cvt};
use crate::io::{default_read, BorrowedCursor, Read, Result};

pub const INSECURE_HASHMAP: bool = false;

pub struct Entropy {
    pub insecure: bool,
}

impl Read for Entropy {
    fn read(&mut self, buf: &mut [u8]) -> Result<usize> {
        default_read(self, buf)
    }

    fn read_buf(&mut self, mut buf: BorrowedCursor<'_>) -> Result<()> {
        unsafe {
            let len = cvt(abi::read_entropy(buf.as_ptr(), buf.capacity(), 0))?;
            buf.advance(len as usize);
            Ok(())
        }
    }
}
