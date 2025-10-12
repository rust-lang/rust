use core::fmt::{Display, Formatter, Result};

#[allow(missing_debug_implementations)]
#[unstable(feature = "ub_checks", issue = "none")]
pub struct DisplayWrapper<T>(#[unstable(feature = "ub_checks", issue = "none")] pub T);

trait Displayable: Sized + Clone + Copy {
    const IS_POINTER: bool = false;
    const IS_INT: bool = false;
    const IS_UINT: bool = false;
    const IS_CHAR: bool = false;
    const IS_STR: bool = false;
    #[inline]
    fn as_char(self) -> char {
        unimplemented!()
    }
    #[inline]
    fn addr(self) -> usize {
        unimplemented!()
    }
    #[inline]
    fn as_u128(self) -> u128 {
        unimplemented!()
    }
    #[inline]
    fn as_i128(self) -> i128 {
        unimplemented!()
    }
}

impl Displayable for char {
    const IS_CHAR: bool = true;
    #[inline]
    fn as_char(self) -> char {
        self
    }
}

impl<T> Displayable for *const T {
    const IS_POINTER: bool = true;
    #[inline]
    fn addr(self) -> usize {
        self.addr()
    }
}
impl<T> Displayable for *mut T {
    const IS_POINTER: bool = true;
    #[inline]
    fn addr(self) -> usize {
        self.addr()
    }
}

impl Displayable for u8 {
    const IS_UINT: bool = true;
    #[inline]
    fn as_u128(self) -> u128 {
        self as u128
    }
}
impl Displayable for u32 {
    const IS_UINT: bool = true;
    #[inline]
    fn as_u128(self) -> u128 {
        self as u128
    }
}
impl Displayable for u64 {
    const IS_UINT: bool = true;
    #[inline]
    fn as_u128(self) -> u128 {
        self as u128
    }
}
impl Displayable for usize {
    const IS_UINT: bool = true;
    #[inline]
    fn as_u128(self) -> u128 {
        self as u128
    }
}
impl Displayable for u128 {
    const IS_UINT: bool = true;
    #[inline]
    fn as_u128(self) -> u128 {
        self
    }
}

impl Displayable for isize {
    const IS_INT: bool = true;
    #[inline]
    fn as_i128(self) -> i128 {
        self as i128
    }
}
impl Displayable for i8 {
    const IS_INT: bool = true;
    #[inline]
    fn as_i128(self) -> i128 {
        self as i128
    }
}
impl Displayable for i16 {
    const IS_INT: bool = true;
    #[inline]
    fn as_i128(self) -> i128 {
        self as i128
    }
}
impl Displayable for i32 {
    const IS_INT: bool = true;
    #[inline]
    fn as_i128(self) -> i128 {
        self as i128
    }
}
impl Displayable for i64 {
    const IS_INT: bool = true;
    #[inline]
    fn as_i128(self) -> i128 {
        self as i128
    }
}
impl Displayable for i128 {
    const IS_INT: bool = true;
    #[inline]
    fn as_i128(self) -> i128 {
        self
    }
}

#[unstable(feature = "ub_checks", issue = "none")]
impl<T: Displayable> Display for DisplayWrapper<T> {
    #[inline]
    fn fmt(&self, f: &mut Formatter<'_>) -> Result {
        const HEX: [u8; 16] = *b"0123456789abcdef";
        assert!(T::IS_POINTER ^ T::IS_UINT ^ T::IS_INT ^ T::IS_CHAR);
        if T::IS_CHAR {
            let mut buf = [0u8; 4];
            let s = self.0.as_char().encode_utf8(&mut buf);
            return f.write_str(s);
        }
        /*
        if T::IS_STR {
            return f.write_str(self.0.as_str());
        }
        */

        let mut buf = [0u8; 42];
        let mut cur = buf.len();

        if T::IS_POINTER {
            let mut n = self.0.addr();
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
        } else {
            let mut is_negative = false;
            let mut n = if T::IS_INT {
                let signed = self.0.as_i128();
                is_negative = signed < 0;
                (!(signed as u128)).wrapping_add(1)
            } else {
                self.0.as_u128()
            };
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
        let s = unsafe { core::str::from_utf8_unchecked(&buf[cur..]) };
        f.write_str(s)
    }
}
