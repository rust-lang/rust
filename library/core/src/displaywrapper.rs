use core::fmt::{Display, Formatter, Result};

#[allow(missing_debug_implementations)]
#[unstable(feature = "ub_checks", issue = "none")]
pub enum DisplayWrapper<'a> {
    Bool(bool),
    Char(char),
    Str(&'a str),
    Ptr(*const ()),
    Uint(u128),
    Int(i128),
}

impl From<bool> for DisplayWrapper<'_> {
    fn from(b: bool) -> Self {
        Self::Bool(b)
    }
}

impl<'a> From<&'a str> for DisplayWrapper<'a> {
    fn from(s: &'a str) -> Self {
        Self::Str(s)
    }
}

impl From<char> for DisplayWrapper<'_> {
    fn from(c: char) -> Self {
        Self::Char(c)
    }
}

impl From<*const ()> for DisplayWrapper<'_> {
    fn from(c: *const ()) -> Self {
        Self::Ptr(c)
    }
}
impl From<*mut ()> for DisplayWrapper<'_> {
    fn from(c: *mut ()) -> Self {
        Self::Ptr(c as *const ())
    }
}

impl From<u8> for DisplayWrapper<'_> {
    fn from(c: u8) -> Self {
        Self::Uint(c as u128)
    }
}
impl From<u16> for DisplayWrapper<'_> {
    fn from(c: u16) -> Self {
        Self::Uint(c as u128)
    }
}
impl From<u32> for DisplayWrapper<'_> {
    fn from(c: u32) -> Self {
        Self::Uint(c as u128)
    }
}
impl From<u64> for DisplayWrapper<'_> {
    fn from(c: u64) -> Self {
        Self::Uint(c as u128)
    }
}
impl From<u128> for DisplayWrapper<'_> {
    fn from(c: u128) -> Self {
        Self::Uint(c as u128)
    }
}
impl From<usize> for DisplayWrapper<'_> {
    fn from(c: usize) -> Self {
        Self::Uint(c as u128)
    }
}

impl From<i8> for DisplayWrapper<'_> {
    fn from(c: i8) -> Self {
        Self::Int(c as i128)
    }
}
impl From<i16> for DisplayWrapper<'_> {
    fn from(c: i16) -> Self {
        Self::Int(c as i128)
    }
}
impl From<i32> for DisplayWrapper<'_> {
    fn from(c: i32) -> Self {
        Self::Int(c as i128)
    }
}
impl From<i64> for DisplayWrapper<'_> {
    fn from(c: i64) -> Self {
        Self::Int(c as i128)
    }
}
impl From<i128> for DisplayWrapper<'_> {
    fn from(c: i128) -> Self {
        Self::Int(c as i128)
    }
}
impl From<isize> for DisplayWrapper<'_> {
    fn from(c: isize) -> Self {
        Self::Int(c as i128)
    }
}

#[unstable(feature = "ub_checks", issue = "none")]
impl Display for DisplayWrapper<'_> {
    #[inline]
    fn fmt(&self, f: &mut Formatter<'_>) -> Result {
        const HEX: [u8; 16] = *b"0123456789abcdef";
        let mut buf = [0u8; 42];
        let mut cur = buf.len();

        match *self {
            Self::Bool(_) | Self::Str(_) => panic!(),
            Self::Char(c) => {
                let mut buf = [0u8; 4];
                let s = c.encode_utf8(&mut buf);
                return f.write_str(s);
            }
            Self::Ptr(ptr) => {
                let mut n = ptr.addr();
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
            }
            Self::Uint(mut n) => {
                while n >= 10 {
                    let d = n % 10;
                    n /= 10;
                    cur -= 1;
                    buf[cur] = (d as u8) + b'0';
                }
                cur -= 1;
                buf[cur] = (n as u8) + b'0';
            }
            Self::Int(n) => {
                let is_negative = n < 0;
                let mut n = (!(n as u128)).wrapping_add(1);

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
            }
        }
        // SAFETY: The buffer is initially ASCII and we only write ASCII bytes to it.
        let s = unsafe { core::str::from_utf8_unchecked(&buf[cur..]) };
        f.write_str(s)
    }
}
