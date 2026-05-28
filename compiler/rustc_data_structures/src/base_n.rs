//! Converts unsigned integers into a string representation with some base.
//! Bases up to and including 36 can be used for case-insensitive things.

use std::{ascii, fmt};

#[cfg(test)]
mod tests;

pub const MAX_BASE: usize = 64;
pub const ALPHANUMERIC_ONLY: usize = 62;
pub const CASE_INSENSITIVE: usize = 36;

const BASE_64: [ascii::Char; MAX_BASE] = {
    let bytes = b"0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ@$";
    let Some(ascii) = bytes.as_ascii() else { panic!() };
    *ascii
};

pub struct BaseNString {
    start: usize,
    buf: [ascii::Char; 128],
}

impl std::ops::Deref for BaseNString {
    type Target = str;

    fn deref(&self) -> &str {
        self.buf[self.start..].as_str()
    }
}

impl AsRef<str> for BaseNString {
    fn as_ref(&self) -> &str {
        self.buf[self.start..].as_str()
    }
}

impl fmt::Display for BaseNString {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self)
    }
}

// This trait just lets us reserve the exact right amount of space when doing fixed-length
// case-insensitive encoding. Add any impls you need.
pub trait ToBaseN: Into<u128> {
    fn encoded_len(base: usize) -> usize;

    fn to_base_fixed_len(self, base: usize) -> BaseNString {
        let mut encoded = self.to_base(base);
        encoded.start = encoded.buf.len() - Self::encoded_len(base);
        encoded
    }

    fn to_base(self, base: usize) -> BaseNString {
        let mut output = [ascii::Char::Digit0; 128];

        let mut n: u128 = self.into();

        let mut index = output.len();
        loop {
            index -= 1;
            output[index] = BASE_64[(n % base as u128) as usize];
            n /= base as u128;

            if n == 0 {
                break;
            }
        }
        assert_eq!(n, 0);

        BaseNString { start: index, buf: output }
    }
}

impl ToBaseN for u128 {
    fn encoded_len(base: usize) -> usize {
        let mut max = u128::MAX;
        let mut len = 0;
        while max > 0 {
            len += 1;
            max /= base as u128;
        }
        len
    }
}

impl ToBaseN for u64 {
    fn encoded_len(base: usize) -> usize {
        let mut max = u64::MAX;
        let mut len = 0;
        while max > 0 {
            len += 1;
            max /= base as u64;
        }
        len
    }
}

impl ToBaseN for u32 {
    fn encoded_len(base: usize) -> usize {
        let mut max = u32::MAX;
        let mut len = 0;
        while max > 0 {
            len += 1;
            max /= base as u32;
        }
        len
    }
}
