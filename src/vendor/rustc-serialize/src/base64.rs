// Copyright 2012-2014 The Rust Project Developers. See the COPYRIGHT
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

//! Base64 binary-to-text encoding

pub use self::FromBase64Error::*;
pub use self::CharacterSet::*;

use std::fmt;
use std::error;

/// Available encoding character sets
#[derive(Clone, Copy, Debug)]
pub enum CharacterSet {
    /// The standard character set (uses `+` and `/`)
    Standard,
    /// The URL safe character set (uses `-` and `_`)
    UrlSafe
}

/// Available newline types
#[derive(Clone, Copy, Debug)]
pub enum Newline {
    /// A linefeed (i.e. Unix-style newline)
    LF,
    /// A carriage return and a linefeed (i.e. Windows-style newline)
    CRLF
}

/// Contains configuration parameters for `to_base64`.
#[derive(Clone, Copy, Debug)]
pub struct Config {
    /// Character set to use
    pub char_set: CharacterSet,
    /// Newline to use
    pub newline: Newline,
    /// True to pad output with `=` characters
    pub pad: bool,
    /// `Some(len)` to wrap lines at `len`, `None` to disable line wrapping
    pub line_length: Option<usize>
}

/// Configuration for RFC 4648 standard base64 encoding
pub static STANDARD: Config =
    Config {char_set: Standard, newline: Newline::CRLF, pad: true, line_length: None};

/// Configuration for RFC 4648 base64url encoding
pub static URL_SAFE: Config =
    Config {char_set: UrlSafe, newline: Newline::CRLF, pad: false, line_length: None};

/// Configuration for RFC 2045 MIME base64 encoding
pub static MIME: Config =
    Config {char_set: Standard, newline: Newline::CRLF, pad: true, line_length: Some(76)};

static STANDARD_CHARS: &'static[u8] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZ\
                                        abcdefghijklmnopqrstuvwxyz\
                                        0123456789+/";

static URLSAFE_CHARS: &'static[u8] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZ\
                                       abcdefghijklmnopqrstuvwxyz\
                                       0123456789-_";

/// A trait for converting a value to base64 encoding.
pub trait ToBase64 {
    /// Converts the value of `self` to a base64 value following the specified
    /// format configuration, returning the owned string.
    fn to_base64(&self, config: Config) -> String;
}

impl ToBase64 for [u8] {
    /// Turn a vector of `u8` bytes into a base64 string.
    ///
    /// # Example
    ///
    /// ```rust
    /// extern crate rustc_serialize;
    /// use rustc_serialize::base64::{ToBase64, STANDARD};
    ///
    /// fn main () {
    ///     let str = [52,32].to_base64(STANDARD);
    ///     println!("base 64 output: {:?}", str);
    /// }
    /// ```
    fn to_base64(&self, config: Config) -> String {
        let bytes = match config.char_set {
            Standard => STANDARD_CHARS,
            UrlSafe => URLSAFE_CHARS
        };

        let len = self.len();
        let newline = match config.newline {
            Newline::LF => "\n",
            Newline::CRLF => "\r\n",
        };

        // Preallocate memory.
        let mut prealloc_len = (len + 2) / 3 * 4;
        if let Some(line_length) = config.line_length {
            let num_lines = match prealloc_len {
                0 => 0,
                n => (n - 1) / line_length
            };
            prealloc_len += num_lines * newline.bytes().count();
        }

        let mut out_bytes = vec![b'='; prealloc_len];

        // Deal with padding bytes
        let mod_len = len % 3;

        // Use iterators to reduce branching
        {
            let mut cur_length = 0;

            let mut s_in = self[..len - mod_len].iter().map(|&x| x as u32);
            let mut s_out = out_bytes.iter_mut();

            // Convenient shorthand
            let enc = |val| bytes[val as usize];
            let mut write = |val| *s_out.next().unwrap() = val;

            // Iterate though blocks of 4
            while let (Some(first), Some(second), Some(third)) =
                        (s_in.next(), s_in.next(), s_in.next()) {

                // Line break if needed
                if let Some(line_length) = config.line_length {
                    if cur_length >= line_length {
                        for b in newline.bytes() { write(b) };
                        cur_length = 0;
                    }
                }

                let n = first << 16 | second << 8 | third;

                // This 24-bit number gets separated into four 6-bit numbers.
                write(enc((n >> 18) & 63));
                write(enc((n >> 12) & 63));
                write(enc((n >> 6 ) & 63));
                write(enc((n >> 0 ) & 63));

                cur_length += 4;
            }

            // Line break only needed if padding is required
            if mod_len != 0 {
                if let Some(line_length) = config.line_length {
                    if cur_length >= line_length {
                        for b in newline.bytes() { write(b) };
                    }
                }
            }

            // Heh, would be cool if we knew this was exhaustive
            // (the dream of bounded integer types)
            match mod_len {
                0 => (),
                1 => {
                    let n = (self[len-1] as u32) << 16;
                    write(enc((n >> 18) & 63));
                    write(enc((n >> 12) & 63));
                }
                2 => {
                    let n = (self[len-2] as u32) << 16 |
                            (self[len-1] as u32) << 8;
                    write(enc((n >> 18) & 63));
                    write(enc((n >> 12) & 63));
                    write(enc((n >> 6 ) & 63));
                }
                _ => panic!("Algebra is broken, please alert the math police")
            }
        }

        // We get padding for "free", so only have to drop it if unwanted.
        if !config.pad {
            while let Some(&b'=') = out_bytes.last() {
                out_bytes.pop();
            }
        }

        unsafe { String::from_utf8_unchecked(out_bytes) }
    }
}

/// A trait for converting from base64 encoded values.
pub trait FromBase64 {
    /// Converts the value of `self`, interpreted as base64 encoded data, into
    /// an owned vector of bytes, returning the vector.
    fn from_base64(&self) -> Result<Vec<u8>, FromBase64Error>;
}

/// Errors that can occur when decoding a base64 encoded string
#[derive(Clone, Copy)]
pub enum FromBase64Error {
    /// The input contained a character not part of the base64 format
    InvalidBase64Byte(u8, usize),
    /// The input had an invalid length
    InvalidBase64Length,
}

impl fmt::Debug for FromBase64Error {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match *self {
            InvalidBase64Byte(ch, idx) =>
                write!(f, "Invalid character '{}' at position {}", ch, idx),
            InvalidBase64Length => write!(f, "Invalid length"),
        }
    }
}

impl error::Error for FromBase64Error {
    fn description(&self) -> &str {
        match *self {
            InvalidBase64Byte(_, _) => "invalid character",
            InvalidBase64Length => "invalid length",
        }
    }
}

impl fmt::Display for FromBase64Error {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        fmt::Debug::fmt(&self, f)
    }
}

impl FromBase64 for str {
    /// Convert any base64 encoded string (literal, `@`, `&`, or `~`)
    /// to the byte values it encodes.
    ///
    /// You can use the `String::from_utf8` function to turn a `Vec<u8>` into a
    /// string with characters corresponding to those values.
    ///
    /// # Example
    ///
    /// This converts a string literal to base64 and back.
    ///
    /// ```rust
    /// extern crate rustc_serialize;
    /// use rustc_serialize::base64::{ToBase64, FromBase64, STANDARD};
    ///
    /// fn main () {
    ///     let hello_str = b"Hello, World".to_base64(STANDARD);
    ///     println!("base64 output: {}", hello_str);
    ///     let res = hello_str.from_base64();
    ///     if res.is_ok() {
    ///       let opt_bytes = String::from_utf8(res.unwrap());
    ///       if opt_bytes.is_ok() {
    ///         println!("decoded from base64: {:?}", opt_bytes.unwrap());
    ///       }
    ///     }
    /// }
    /// ```
    #[inline]
    fn from_base64(&self) -> Result<Vec<u8>, FromBase64Error> {
        self.as_bytes().from_base64()
    }
}

impl FromBase64 for [u8] {
    fn from_base64(&self) -> Result<Vec<u8>, FromBase64Error> {
        let mut r = Vec::with_capacity(self.len());
        let mut buf: u32 = 0;
        let mut modulus = 0;

        let mut it = self.iter();
        for byte in it.by_ref() {
            let code = DECODE_TABLE[*byte as usize];
            if code >= SPECIAL_CODES_START {
                match code {
                    NEWLINE_CODE => continue,
                    EQUALS_CODE => break,
                    INVALID_CODE => return Err(InvalidBase64Byte(
                            *byte, (byte as *const _ as usize) - self.as_ptr() as usize)),
                    _ => unreachable!(),
                }
            }
            buf = (buf | code as u32) << 6;
            modulus += 1;
            if modulus == 4 {
                modulus = 0;
                r.push((buf >> 22) as u8);
                r.push((buf >> 14) as u8);
                r.push((buf >> 6 ) as u8);
            }
        }

        for byte in it {
            match *byte {
                b'=' | b'\r' | b'\n' => continue,
                _ => return Err(InvalidBase64Byte(
                        *byte, (byte as *const _ as usize) - self.as_ptr() as usize)),
            }
        }

        match modulus {
            2 => {
                r.push((buf >> 10) as u8);
            }
            3 => {
                r.push((buf >> 16) as u8);
                r.push((buf >> 8 ) as u8);
            }
            0 => (),
            _ => return Err(InvalidBase64Length),
        }

        Ok(r)
    }
}

/// Base64 decoding lookup table, generated using:
/// ```rust
///     let mut ch = 0u8;
///     loop {
///         let code = match ch {
///             b'A'...b'Z' => ch - 0x41,
///             b'a'...b'z' => ch - 0x47,
///             b'0'...b'9' => ch + 0x04,
///             b'+' | b'-' => 0x3E,
///             b'/' | b'_' => 0x3F,
///             b'=' => 0xFE,
///             b'\r' | b'\n' => 0xFD,
///             _ => 0xFF,
///         };
///         print!("0x{:02X}, ", code);
///         if ch % 16  == 15 { println!(""); }
///         else if ch == 0xFF { break; }
///         ch += 1;
///     }
///     println!("");
/// }
/// ```
const DECODE_TABLE: [u8; 256] = [
    0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFD, 0xFF, 0xFF, 0xFD, 0xFF, 0xFF,
    0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF,
    0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0x3E, 0xFF, 0x3E, 0xFF, 0x3F,
    0x34, 0x35, 0x36, 0x37, 0x38, 0x39, 0x3A, 0x3B, 0x3C, 0x3D, 0xFF, 0xFF, 0xFF, 0xFE, 0xFF, 0xFF,
    0xFF, 0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0A, 0x0B, 0x0C, 0x0D, 0x0E,
    0x0F, 0x10, 0x11, 0x12, 0x13, 0x14, 0x15, 0x16, 0x17, 0x18, 0x19, 0xFF, 0xFF, 0xFF, 0xFF, 0x3F,
    0xFF, 0x1A, 0x1B, 0x1C, 0x1D, 0x1E, 0x1F, 0x20, 0x21, 0x22, 0x23, 0x24, 0x25, 0x26, 0x27, 0x28,
    0x29, 0x2A, 0x2B, 0x2C, 0x2D, 0x2E, 0x2F, 0x30, 0x31, 0x32, 0x33, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF,
    0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF,
    0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF,
    0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF,
    0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF,
    0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF,
    0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF,
    0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF,
    0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF,
];
const INVALID_CODE: u8 = 0xFF;
const EQUALS_CODE: u8 = 0xFE;
const NEWLINE_CODE: u8 = 0xFD;
const SPECIAL_CODES_START: u8 = NEWLINE_CODE;

#[cfg(test)]
mod tests {
    use base64::{Config, Newline, FromBase64, ToBase64, STANDARD, URL_SAFE};

    #[test]
    fn test_to_base64_basic() {
        assert_eq!("".as_bytes().to_base64(STANDARD), "");
        assert_eq!("f".as_bytes().to_base64(STANDARD), "Zg==");
        assert_eq!("fo".as_bytes().to_base64(STANDARD), "Zm8=");
        assert_eq!("foo".as_bytes().to_base64(STANDARD), "Zm9v");
        assert_eq!("foob".as_bytes().to_base64(STANDARD), "Zm9vYg==");
        assert_eq!("fooba".as_bytes().to_base64(STANDARD), "Zm9vYmE=");
        assert_eq!("foobar".as_bytes().to_base64(STANDARD), "Zm9vYmFy");
    }

    #[test]
    fn test_to_base64_crlf_line_break() {
        assert!(![08; 1000].to_base64(Config {line_length: None, ..STANDARD})
                              .contains("\r\n"));
        assert_eq!(b"foobar".to_base64(Config {line_length: Some(4),
                                               ..STANDARD}),
                   "Zm9v\r\nYmFy");
    }

    #[test]
    fn test_to_base64_lf_line_break() {
        assert!(![08; 1000].to_base64(Config {line_length: None,
                                                 newline: Newline::LF,
                                                 ..STANDARD})
                              .contains("\n"));
        assert_eq!(b"foobar".to_base64(Config {line_length: Some(4),
                                               newline: Newline::LF,
                                               ..STANDARD}),
                   "Zm9v\nYmFy");
    }

    #[test]
    fn test_to_base64_padding() {
        assert_eq!("f".as_bytes().to_base64(Config {pad: false, ..STANDARD}), "Zg");
        assert_eq!("fo".as_bytes().to_base64(Config {pad: false, ..STANDARD}), "Zm8");
    }

    #[test]
    fn test_to_base64_url_safe() {
        assert_eq!([251, 255].to_base64(URL_SAFE), "-_8");
        assert_eq!([251, 255].to_base64(STANDARD), "+/8=");
    }

    #[test]
    fn test_to_base64_empty_line_length() {
        [].to_base64(Config {line_length: Some(72), ..STANDARD});
    }

    #[test]
    fn test_from_base64_basic() {
        assert_eq!("".from_base64().unwrap(), b"");
        assert_eq!("Zg==".from_base64().unwrap(), b"f");
        assert_eq!("Zm8=".from_base64().unwrap(), b"fo");
        assert_eq!("Zm9v".from_base64().unwrap(), b"foo");
        assert_eq!("Zm9vYg==".from_base64().unwrap(), b"foob");
        assert_eq!("Zm9vYmE=".from_base64().unwrap(), b"fooba");
        assert_eq!("Zm9vYmFy".from_base64().unwrap(), b"foobar");
    }

    #[test]
    fn test_from_base64_bytes() {
        assert_eq!(b"Zm9vYmFy".from_base64().unwrap(), b"foobar");
    }

    #[test]
    fn test_from_base64_newlines() {
        assert_eq!("Zm9v\r\nYmFy".from_base64().unwrap(),
                   b"foobar");
        assert_eq!("Zm9vYg==\r\n".from_base64().unwrap(),
                   b"foob");
        assert_eq!("Zm9v\nYmFy".from_base64().unwrap(),
                   b"foobar");
        assert_eq!("Zm9vYg==\n".from_base64().unwrap(),
                   b"foob");
    }

    #[test]
    fn test_from_base64_urlsafe() {
        assert_eq!("-_8".from_base64().unwrap(), "+/8=".from_base64().unwrap());
    }

    #[test]
    fn test_from_base64_invalid_char() {
        assert!("Zm$=".from_base64().is_err());
        assert!("Zg==$".from_base64().is_err());
    }

    #[test]
    fn test_from_base64_invalid_padding() {
        assert!("Z===".from_base64().is_err());
    }

    #[test]
    fn test_base64_random() {
        use rand::{thread_rng, Rng};

        for _ in 0..1000 {
            let times = thread_rng().gen_range(1, 100);
            let v = thread_rng().gen_iter::<u8>().take(times)
                                .collect::<Vec<_>>();
            assert_eq!(v.to_base64(STANDARD)
                        .from_base64()
                        .unwrap(),
                       v);
        }
    }
}
