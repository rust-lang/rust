use crate::fmt;
use crate::mem::MaybeUninit;
use crate::str;

/// Used for slow path in `Display` implementations when alignment is required.
pub struct DisplayBuffer<const SIZE: usize> {
    buf: [MaybeUninit<u8>; SIZE],
    len: usize,
}

impl<const SIZE: usize> DisplayBuffer<SIZE> {
    #[inline]
    pub const fn new() -> Self {
        Self { buf: MaybeUninit::uninit_array(), len: 0 }
    }

    #[inline]
    pub fn as_str(&self) -> &str {
        // SAFETY: `buf` is only written to by the `fmt::Write::write_str` implementation
        // which writes a valid UTF-8 string to `buf` and correctly sets `len`.
        unsafe {
            let s = MaybeUninit::slice_assume_init_ref(&self.buf[..self.len]);
            str::from_utf8_unchecked(s)
        }
    }
}

impl<const SIZE: usize> fmt::Write for DisplayBuffer<SIZE> {
    fn write_str(&mut self, s: &str) -> fmt::Result {
        let bytes = s.as_bytes();

        if let Some(buf) = self.buf.get_mut(self.len..(self.len + bytes.len())) {
            MaybeUninit::write_slice(buf, bytes);
            self.len += bytes.len();
            Ok(())
        } else {
            Err(fmt::Error)
        }
    }
}
