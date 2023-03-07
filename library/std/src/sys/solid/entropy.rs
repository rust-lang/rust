use super::abi;
use super::error::SolidError;
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
        SolidError::err_if_negative(unsafe {
            abi::SOLID_RNG_SampleRandomBytes(buf.as_mut().as_mut_ptr().cast(), buf.capacity())
        })
        .map_err(|e| e.as_io_error())?;

        unsafe {
            buf.advance(buf.capacity());
            Ok(())
        }
    }

    fn read_buf_exact(&mut self, buf: BorrowedCursor<'_>) -> Result<()> {
        self.read_buf(buf)
    }
}
