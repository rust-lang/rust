//! Hex binary-to-text encoding

pub use self::FromHexError::*;

use std::fmt;
use std::error;

/// A trait for converting a value to hexadecimal encoding
pub trait ToHex {
    /// Converts the value of `self` to a hex value, returning the owned
    /// string.
    fn to_hex(&self) -> String;
}

const CHARS: &[u8] = b"0123456789abcdef";

impl ToHex for [u8] {
    /// Turn a vector of `u8` bytes into a hexadecimal string.
    ///
    /// # Examples
    ///
    /// ```
    /// #![feature(rustc_private)]
    ///
    /// extern crate serialize;
    /// use serialize::hex::ToHex;
    ///
    /// fn main () {
    ///     let str = [52,32].to_hex();
    ///     println!("{}", str);
    /// }
    /// ```
    fn to_hex(&self) -> String {
        let mut v = Vec::with_capacity(self.len() * 2);
        for &byte in self {
            v.push(CHARS[(byte >> 4) as usize]);
            v.push(CHARS[(byte & 0xf) as usize]);
        }

        unsafe {
            String::from_utf8_unchecked(v)
        }
    }
}

/// A trait for converting hexadecimal encoded values
pub trait FromHex {
    /// Converts the value of `self`, interpreted as hexadecimal encoded data,
    /// into an owned vector of bytes, returning the vector.
    fn from_hex(&self) -> Result<Vec<u8>, FromHexError>;
}

/// Errors that can occur when decoding a hex encoded string
#[derive(Copy, Clone, Debug)]
pub enum FromHexError {
    /// The input contained a character not part of the hex format
    InvalidHexCharacter(char, usize),
    /// The input had an invalid length
    InvalidHexLength,
}

impl fmt::Display for FromHexError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match *self {
            InvalidHexCharacter(ch, idx) =>
                write!(f, "Invalid character '{}' at position {}", ch, idx),
            InvalidHexLength => write!(f, "Invalid input length"),
        }
    }
}

impl error::Error for FromHexError {
    fn description(&self) -> &str {
        match *self {
            InvalidHexCharacter(..) => "invalid character",
            InvalidHexLength => "invalid length",
        }
    }
}


impl FromHex for str {
    /// Converts any hexadecimal encoded string (literal, `@`, `&`, or `~`)
    /// to the byte values it encodes.
    ///
    /// You can use the `String::from_utf8` function to turn a
    /// `Vec<u8>` into a string with characters corresponding to those values.
    ///
    /// # Examples
    ///
    /// This converts a string literal to hexadecimal and back.
    ///
    /// ```
    /// #![feature(rustc_private)]
    ///
    /// extern crate serialize;
    /// use serialize::hex::{FromHex, ToHex};
    ///
    /// fn main () {
    ///     let hello_str = "Hello, World".as_bytes().to_hex();
    ///     println!("{}", hello_str);
    ///     let bytes = hello_str.from_hex().unwrap();
    ///     println!("{:?}", bytes);
    ///     let result_str = String::from_utf8(bytes).unwrap();
    ///     println!("{}", result_str);
    /// }
    /// ```
    fn from_hex(&self) -> Result<Vec<u8>, FromHexError> {
        // This may be an overestimate if there is any whitespace
        let mut b = Vec::with_capacity(self.len() / 2);
        let mut modulus = 0;
        let mut buf = 0;

        for (idx, byte) in self.bytes().enumerate() {
            buf <<= 4;

            match byte {
                b'A'..=b'F' => buf |= byte - b'A' + 10,
                b'a'..=b'f' => buf |= byte - b'a' + 10,
                b'0'..=b'9' => buf |= byte - b'0',
                b' '|b'\r'|b'\n'|b'\t' => {
                    buf >>= 4;
                    continue
                }
                _ => {
                    let ch = self[idx..].chars().next().unwrap();
                    return Err(InvalidHexCharacter(ch, idx))
                }
            }

            modulus += 1;
            if modulus == 2 {
                modulus = 0;
                b.push(buf);
            }
        }

        match modulus {
            0 => Ok(b),
            _ => Err(InvalidHexLength),
        }
    }
}

#[cfg(test)]
mod tests;
