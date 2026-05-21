#[cfg(test)]
mod tests;

#[stable(feature = "rust1", since = "1.0.0")]
pub use alloc::io::Cursor;

use crate::io::{self, BufRead};

#[stable(feature = "rust1", since = "1.0.0")]
impl<T> BufRead for Cursor<T>
where
    T: AsRef<[u8]>,
{
    fn fill_buf(&mut self) -> io::Result<&[u8]> {
        Ok(Cursor::split(self).1)
    }
    fn consume(&mut self, amt: usize) {
        self.set_position(self.position() + amt as u64);
    }
}
