use crate::io::{const_io_error, default_read, BorrowedCursor, ErrorKind, Read, Result};

pub const INSECURE_HASHMAP: bool = false;

pub struct Entropy {
    pub insecure: bool,
}

impl Read for Entropy {
    fn read(&mut self, buf: &mut [u8]) -> Result<usize> {
        default_read(self, buf)
    }

    fn read_buf(&mut self, mut buf: BorrowedCursor<'_>) -> Result<()> {
        if buf.capacity() != 0 {
            let rand = rdrand64()
                .ok_or(const_io_error!(ErrorKind::WouldBlock, "no random data available"))?;
            buf.append(&rand.to_ne_bytes()[..usize::min(buf.capacity(), 8)]);
            Ok(())
        } else {
            Ok(())
        }
    }
}

pub(super) fn rdrand64() -> Option<u64> {
    unsafe {
        let mut ret: u64 = 0;
        for _ in 0..10 {
            if crate::arch::x86_64::_rdrand64_step(&mut ret) == 1 {
                return Some(ret);
            }
        }
        None
    }
}
