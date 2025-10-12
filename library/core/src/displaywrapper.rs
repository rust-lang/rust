use core::fmt::{Display, Formatter, Result};

#[allow(missing_debug_implementations)]
pub struct DisplayWrapper<T>(pub T);

macro_rules! display_int {
    ($ty:ty) => {
        impl Display for DisplayWrapper<$ty> {
            #[inline]
            fn fmt(&self, f: &mut Formatter<'_>) -> Result {
                let n = self.0;
                let is_negative = n < 0;
                let n = (!(n as u128)).wrapping_add(1);
                display_int(n, is_negative, f)
            }
        }
    };
}

display_int!(i8);
display_int!(i16);
display_int!(i32);
display_int!(i64);
display_int!(i128);
display_int!(isize);

macro_rules! display_uint {
    ($ty:ty) => {
        impl Display for DisplayWrapper<$ty> {
            #[inline]
            fn fmt(&self, f: &mut Formatter<'_>) -> Result {
                display_int(self.0 as u128, false, f)
            }
        }
    };
}

display_uint!(u8);
display_uint!(u16);
display_uint!(u32);
display_uint!(u64);
display_uint!(u128);
display_uint!(usize);

impl Display for DisplayWrapper<*const ()> {
    #[inline]
    fn fmt(&self, f: &mut Formatter<'_>) -> Result {
        format_ptr(self.0.addr(), f)
    }
}
impl Display for DisplayWrapper<*mut ()> {
    #[inline]
    fn fmt(&self, f: &mut Formatter<'_>) -> Result {
        format_ptr(self.0.addr(), f)
    }
}

#[inline]
fn format_ptr(addr: usize, f: &mut Formatter<'_>) -> Result {
    const HEX: [u8; 16] = *b"0123456789abcdef";
    let mut buf = [0u8; 42];
    let mut cur = buf.len();

    let mut n = addr;
    while n >= 16 {
        let d = n % 16;
        n /= 16;
        cur -= 1;
        buf[cur] = HEX[d];
    }
    cur -= 1;
    buf[cur] = HEX[n];

    cur -= 1;
    buf[cur] = b'x';
    cur -= 1;
    buf[cur] = b'0';

    // SAFETY: The buffer is initially ASCII and we only write ASCII bytes to it.
    let s = unsafe { core::str::from_utf8_unchecked(&buf[cur..]) };
    f.write_str(s)
}

impl Display for DisplayWrapper<char> {
    #[inline]
    fn fmt(&self, f: &mut Formatter<'_>) -> Result {
        let mut buf = [0u8; 4];
        let s = self.0.encode_utf8(&mut buf);
        f.write_str(s)
    }
}

impl Display for DisplayWrapper<bool> {
    #[inline]
    fn fmt(&self, f: &mut Formatter<'_>) -> Result {
        let s = match self.0 {
            true => "true",
            false => "false",
        };
        f.write_str(s)
    }
}

#[inline]
fn display_int(mut n: u128, is_negative: bool, f: &mut Formatter<'_>) -> Result {
    let mut buf = [0u8; 42];
    let mut cur = buf.len();

    while n >= 10 {
        let d = n % 10;
        n /= 10;
        cur -= 1;
        buf[cur] = (d as u8) + b'0';
    }
    cur -= 1;
    buf[cur] = (n as u8) + b'0';
    if is_negative {
        cur -= 1;
        buf[cur] = b'-';
    }
    // SAFETY: The buffer is initially ASCII and we only write ASCII bytes to it.
    let s = unsafe { core::str::from_utf8_unchecked(&buf[cur..]) };
    f.write_str(s)
}
