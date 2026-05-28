use crate::mem::MaybeUninit;
use crate::{fmt, str};

/// Used for slow path in `Display` implementations when alignment is required.
pub(super) struct DisplayBuffer<const SIZE: usize> {
    buf: [MaybeUninit<u8>; SIZE],
    len: usize,
}

impl<const SIZE: usize> DisplayBuffer<SIZE> {
    #[inline]
    pub(super) const fn new() -> Self {
        Self { buf: [MaybeUninit::uninit(); SIZE], len: 0 }
    }

    #[inline]
    pub(super) fn as_str(&self) -> &str {
        // SAFETY: `buf` is only written to by the `fmt::Write::write_str` implementation
        // which writes a valid UTF-8 string to `buf` and correctly sets `len`.
        unsafe {
            let s = self.buf[..self.len].assume_init_ref();
            str::from_utf8_unchecked(s)
        }
    }
}

impl<const SIZE: usize> fmt::Write for DisplayBuffer<SIZE> {
    fn write_str(&mut self, s: &str) -> fmt::Result {
        let bytes = s.as_bytes();

        if let Some(buf) = self.buf.get_mut(self.len..(self.len + bytes.len())) {
            buf.write_copy_of_slice(bytes);
            self.len += bytes.len();
            Ok(())
        } else {
            Err(fmt::Error)
        }
    }
}
