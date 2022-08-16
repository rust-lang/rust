use crate::fmt;
use crate::str;

/// Used for slow path in `Display` implementations when alignment is required.
pub struct IpDisplayBuffer<const SIZE: usize> {
    buf: [u8; SIZE],
    len: usize,
}

impl<const SIZE: usize> IpDisplayBuffer<SIZE> {
    #[inline(always)]
    pub const fn new(_ip: &[u8; SIZE]) -> Self {
        Self { buf: [0; SIZE], len: 0 }
    }

    #[inline(always)]
    pub fn as_str(&self) -> &str {
        // SAFETY: `buf` is only written to by the `fmt::Write::write_str` implementation
        // which writes a valid UTF-8 string to `buf` and correctly sets `len`.
        unsafe { str::from_utf8_unchecked(&self.buf[..self.len]) }
    }
}

impl<const SIZE: usize> fmt::Write for IpDisplayBuffer<SIZE> {
    fn write_str(&mut self, s: &str) -> fmt::Result {
        if let Some(buf) = self.buf.get_mut(self.len..(self.len + s.len())) {
            buf.copy_from_slice(s.as_bytes());
            self.len += s.len();
            Ok(())
        } else {
            Err(fmt::Error)
        }
    }
}
