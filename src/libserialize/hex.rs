// Copyright 2013-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.
//
// ignore-lexer-test FIXME #15679

//! Hex binary-to-text encoding
use std::fmt;
use std::string;

/// A trait for converting a value to hexadecimal encoding
pub trait ToHex {
    /// Converts the value of `self` to a hex value, returning the owned
    /// string.
    fn to_hex(&self) -> String;
}

static CHARS: &'static[u8] = b"0123456789abcdef";

impl<'a> ToHex for &'a [u8] {
    /**
     * Turn a vector of `u8` bytes into a hexadecimal string.
     *
     * # Example
     *
     * ```rust
     * extern crate serialize;
     * use serialize::hex::ToHex;
     *
     * fn main () {
     *     let str = [52,32].to_hex();
     *     println!("{}", str);
     * }
     * ```
     */
    fn to_hex(&self) -> String {
        let mut v = Vec::with_capacity(self.len() * 2);
        for &byte in self.iter() {
            v.push(CHARS[(byte >> 4) as uint]);
            v.push(CHARS[(byte & 0xf) as uint]);
        }

        unsafe {
            string::raw::from_utf8(v)
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
pub enum FromHexError {
    /// The input contained a character not part of the hex format
    InvalidHexCharacter(char, uint),
    /// The input had an invalid length
    InvalidHexLength,
}

impl fmt::Show for FromHexError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match *self {
            InvalidHexCharacter(ch, idx) =>
                write!(f, "Invalid character '{}' at position {}", ch, idx),
            InvalidHexLength => write!(f, "Invalid input length"),
        }
    }
}

impl<'a> FromHex for &'a str {
    /**
     * Convert any hexadecimal encoded string (literal, `@`, `&`, or `~`)
     * to the byte values it encodes.
     *
     * You can use the `String::from_utf8` function in `std::string` to turn a
     * `Vec<u8>` into a string with characters corresponding to those values.
     *
     * # Example
     *
     * This converts a string literal to hexadecimal and back.
     *
     * ```rust
     * extern crate serialize;
     * use serialize::hex::{FromHex, ToHex};
     *
     * fn main () {
     *     let hello_str = "Hello, World".as_bytes().to_hex();
     *     println!("{}", hello_str);
     *     let bytes = hello_str.as_slice().from_hex().unwrap();
     *     println!("{}", bytes);
     *     let result_str = String::from_utf8(bytes).unwrap();
     *     println!("{}", result_str);
     * }
     * ```
     */
    fn from_hex(&self) -> Result<Vec<u8>, FromHexError> {
        // This may be an overestimate if there is any whitespace
        let mut b = Vec::with_capacity(self.len() / 2);
        let mut modulus = 0i;
        let mut buf = 0u8;

        for (idx, byte) in self.bytes().enumerate() {
            buf <<= 4;

            match byte {
                b'A'..b'F' => buf |= byte - b'A' + 10,
                b'a'..b'f' => buf |= byte - b'a' + 10,
                b'0'..b'9' => buf |= byte - b'0',
                b' '|b'\r'|b'\n'|b'\t' => {
                    buf >>= 4;
                    continue
                }
                _ => return Err(InvalidHexCharacter(self.char_at(idx), idx)),
            }

            modulus += 1;
            if modulus == 2 {
                modulus = 0;
                b.push(buf);
            }
        }

        match modulus {
            0 => Ok(b.move_iter().collect()),
            _ => Err(InvalidHexLength),
        }
    }
}

#[cfg(test)]
mod tests {
    extern crate test;
    use self::test::Bencher;
    use hex::{FromHex, ToHex};

    #[test]
    pub fn test_to_hex() {
        assert_eq!("foobar".as_bytes().to_hex(), "666f6f626172".to_string());
    }

    #[test]
    pub fn test_from_hex_okay() {
        assert_eq!("666f6f626172".from_hex().unwrap().as_slice(),
                   "foobar".as_bytes());
        assert_eq!("666F6F626172".from_hex().unwrap().as_slice(),
                   "foobar".as_bytes());
    }

    #[test]
    pub fn test_from_hex_odd_len() {
        assert!("666".from_hex().is_err());
        assert!("66 6".from_hex().is_err());
    }

    #[test]
    pub fn test_from_hex_invalid_char() {
        assert!("66y6".from_hex().is_err());
    }

    #[test]
    pub fn test_from_hex_ignores_whitespace() {
        assert_eq!("666f 6f6\r\n26172 ".from_hex().unwrap().as_slice(),
                   "foobar".as_bytes());
    }

    #[test]
    pub fn test_to_hex_all_bytes() {
        for i in range(0u, 256) {
            assert_eq!([i as u8].to_hex(), format!("{:02x}", i as uint));
        }
    }

    #[test]
    pub fn test_from_hex_all_bytes() {
        for i in range(0u, 256) {
            assert_eq!(format!("{:02x}", i as uint).as_slice()
                                                   .from_hex()
                                                   .unwrap()
                                                   .as_slice(),
                       &[i as u8]);
            assert_eq!(format!("{:02X}", i as uint).as_slice()
                                                   .from_hex()
                                                   .unwrap()
                                                   .as_slice(),
                       &[i as u8]);
        }
    }

    #[bench]
    pub fn bench_to_hex(b: &mut Bencher) {
        let s = "イロハニホヘト チリヌルヲ ワカヨタレソ ツネナラム \
                 ウヰノオクヤマ ケフコエテ アサキユメミシ ヱヒモセスン";
        b.iter(|| {
            s.as_bytes().to_hex();
        });
        b.bytes = s.len() as u64;
    }

    #[bench]
    pub fn bench_from_hex(b: &mut Bencher) {
        let s = "イロハニホヘト チリヌルヲ ワカヨタレソ ツネナラム \
                 ウヰノオクヤマ ケフコエテ アサキユメミシ ヱヒモセスン";
        let sb = s.as_bytes().to_hex();
        b.iter(|| {
            sb.as_slice().from_hex().unwrap();
        });
        b.bytes = sb.len() as u64;
    }
}
