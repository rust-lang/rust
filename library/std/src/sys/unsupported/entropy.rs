use super::unsupported;
use crate::io::{BorrowedCursor, Read, Result};

pub const INSECURE_HASHMAP: bool = true;

pub struct Entropy {
    pub insecure: bool,
}

impl Read for Entropy {
    fn read(&mut self, _: &mut [u8]) -> Result<usize> {
        unsupported()
    }

    fn read_buf(&mut self, _: BorrowedCursor<'_>) -> Result<()> {
        unsupported()
    }

    fn read_buf_exact(&mut self, _: BorrowedCursor<'_>) -> Result<()> {
        unsupported()
    }
}
